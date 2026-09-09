package clustering.benchmark.metrics;

import java.lang.management.BufferPoolMXBean;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;
import java.util.ArrayList;
import java.util.List;

/** Samples THIS process — the one running {@code main()}, i.e. the driver — the way
 *  {@link clustering.metrics.FileMetricReporter} samples a Flink JVM.
 *
 *  <h3>Why the driver cannot come from the metrics file any more (5.09.2026)</h3>
 *  It used to: under standalone Application Mode {@code main()} ran INSIDE the JobManager, so the
 *  reporter's {@code .jobmanager.}-scoped lines WERE the driver's counters. The cluster bootstrap
 *  moved to SESSION mode, because Application Mode ships the application jar to TaskManagers as a
 *  PATH ({@code usrlib}) and a remote TaskManager cannot resolve the capturing class of a
 *  serialized lambda from it — every multi-node run died in task deployment
 *  ({@code SerializedLambda} → {@code ClassCastException}; 4 nodes, 18/18 failed). In session mode
 *  the {@code flink run} client uploads the jar to the BlobServer and each TaskManager fetches
 *  that blob into its own user classloader, which is the path that works.
 *
 *  <p>The consequence for measurement: {@code main()} now runs in the CLIENT JVM, which is not a
 *  Flink process at all and therefore has no reporter in it. The JobManager still publishes
 *  {@code .jobmanager.} metrics, but it only coordinates — charging its CPU to
 *  {@code driverCpuTimeNs} would report the wrong process. This class reads the right one.
 *
 *  <p>It also brings Flink CLOSER to Spark rather than further away: Spark runs a client-mode
 *  driver, a separate process sampled by {@code ProcessCpuPlugin}'s driver leg. After this change
 *  both engines' {@code driver*} columns describe the same thing — a separate client JVM running
 *  the algorithm's driver-local code — instead of "the JobManager, which happens to be the driver"
 *  against "a driver process".
 *
 *  <h3>Same readings, same rules</h3>
 *  Deliberately the same MXBeans and the same conventions as the reporter and as the Spark plugin,
 *  because these numbers land in the same columns:
 *  <ul>
 *    <li>CPU from {@code getProcessCpuTime()} — whole process, cumulative since JVM start, so the
 *        run window is a DELTA taken by {@link BenchmarkListener} exactly as for a TaskManager;</li>
 *    <li>GC as the sum of {@link GarbageCollectorMXBean#getCollectionTime()} across collectors,
 *        whole-JVM and never per-task;</li>
 *    <li>heap as a byte-SECONDS integral accumulated per tick, which is what
 *        {@code memoryGbHours} divides;</li>
 *    <li>the heap peak read from {@link MemoryPoolMXBean#getPeakUsage()} and RESET every tick, so
 *        each tick reports its own high-water mark. Ticks tile wall clock, so the max over all of
 *        them is the run's peak and the max over a phase's ticks is that phase's — the trick that
 *        makes a peak sliceable at all (a record book cannot be differenced like an odometer);</li>
 *    <li>direct memory from {@link BufferPoolMXBean}, sampled rather than peaked, because the JDK
 *        exposes no peak API for buffer pools — the same acknowledged blind spot as on the worker
 *        side and on Spark.</li>
 *  </ul>
 *  Sampling period matches the reporter's 200 ms, so a phase boundary is equally sharp on both
 *  legs. */
public final class DriverProcess {

    /** Matches {@code metrics.reporter.file.interval} and Spark's sampling interval. */
    private static final long PERIOD_MS = 200L;

    /** One synthetic process id, so the driver maps carry exactly one entry and
     *  {@link BenchmarkListener}'s per-process delta arithmetic applies unchanged. */
    static final String PROCESS_ID = "client";

    private final List<MemoryPoolMXBean> heapPools = new ArrayList<>();
    private final List<BufferPoolMXBean> directPools = new ArrayList<>();
    private final com.sun.management.OperatingSystemMXBean os;

    /** Per-tick samples: {timestampMs, that tick's heap high-water mark, direct bytes now}. */
    private final List<long[]> series = new ArrayList<>();

    private volatile long cpuNanos;
    private volatile long gcTimeMs;
    private volatile long heapByteSeconds;
    private volatile long peakHeapBytes;
    private volatile long peakDirectBytes;

    private Thread sampler;
    private volatile boolean running;

    public DriverProcess() {
        java.lang.management.OperatingSystemMXBean bean = ManagementFactory.getOperatingSystemMXBean();
        this.os = bean instanceof com.sun.management.OperatingSystemMXBean
            ? (com.sun.management.OperatingSystemMXBean) bean : null;
        for (MemoryPoolMXBean pool : ManagementFactory.getMemoryPoolMXBeans()) {
            if (pool.getType() == MemoryType.HEAP) {
                try {
                    pool.resetPeakUsage();
                    heapPools.add(pool);
                } catch (UnsupportedOperationException ignore) {
                    // a pool that refuses a reset cannot report a per-tick peak; skip it rather
                    // than report a lifetime maximum as if it were this tick's
                }
            }
        }
        for (BufferPoolMXBean pool : ManagementFactory.getPlatformMXBeans(BufferPoolMXBean.class)) {
            if ("direct".equals(pool.getName())) {
                directPools.add(pool);
            }
        }
    }

    /** Starts sampling on a daemon thread. Idempotent. */
    public synchronized void start() {
        if (running) {
            return;
        }
        running = true;
        sample();  // one reading immediately, so a very short run still has a sample
        sampler = new Thread(() -> {
            while (running) {
                try {
                    Thread.sleep(PERIOD_MS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
                sample();
            }
        }, "driver-process-sampler");
        sampler.setDaemon(true);
        sampler.start();
    }

    public synchronized void stop() {
        running = false;
        if (sampler != null) {
            sampler.interrupt();
            sampler = null;
        }
        sample();  // final reading, so the tail of the run is not lost
    }

    private synchronized void sample() {
        if (os != null) {
            long cpu = os.getProcessCpuTime();
            if (cpu > 0) {
                cpuNanos = cpu;
            }
        }
        long gc = 0L;
        for (GarbageCollectorMXBean bean : ManagementFactory.getGarbageCollectorMXBeans()) {
            long t = bean.getCollectionTime();
            if (t > 0) {
                gc += t;
            }
        }
        gcTimeMs = gc;

        long heapUsed = 0L;
        for (MemoryPoolMXBean pool : heapPools) {
            heapUsed += pool.getUsage().getUsed();
        }
        // Integral, not a level: byte-seconds accrued over this tick. PERIOD_MS is used rather
        // than the measured gap so a scheduling hiccup cannot inflate the integral.
        heapByteSeconds += heapUsed * PERIOD_MS / 1000L;

        // Read AND reset, pool by pool, so this tick's figure covers this tick alone.
        long tickPeak = 0L;
        for (MemoryPoolMXBean pool : heapPools) {
            tickPeak += pool.getPeakUsage().getUsed();
            pool.resetPeakUsage();
        }
        peakHeapBytes = Math.max(peakHeapBytes, tickPeak);

        long direct = 0L;
        for (BufferPoolMXBean pool : directPools) {
            direct += pool.getMemoryUsed();
        }
        peakDirectBytes = Math.max(peakDirectBytes, direct);

        series.add(new long[] {System.currentTimeMillis(), tickPeak, direct});
    }

    public synchronized long cpuNanos() {
        return cpuNanos;
    }

    public synchronized long gcTimeMs() {
        return gcTimeMs;
    }

    public synchronized long heapByteSeconds() {
        return heapByteSeconds;
    }

    public synchronized long peakHeapBytes() {
        return peakHeapBytes;
    }

    public synchronized long peakDirectBytes() {
        return peakDirectBytes;
    }

    /** Heap peak within a wall-clock window: the max over the ticks that fall inside it. Gap-free,
     *  because every tick's own high-water mark is recorded and the ticks tile the window. */
    public synchronized long windowHeapPeak(long fromMs, long toMs) {
        long max = 0L;
        for (long[] s : series) {
            if (s[0] >= fromMs && s[0] <= toMs) {
                max = Math.max(max, s[1]);
            }
        }
        return max;
    }

    /** Direct memory within a window: a max of SAMPLES, so a spike between two ticks is invisible
     *  here — the buffer-pool blind spot noted in the class doc. */
    public synchronized long windowDirectPeak(long fromMs, long toMs) {
        long max = 0L;
        for (long[] s : series) {
            if (s[0] >= fromMs && s[0] <= toMs) {
                max = Math.max(max, s[2]);
            }
        }
        return max;
    }
}
