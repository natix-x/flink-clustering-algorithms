package clustering.benchmark.metrics;

import java.lang.management.BufferPoolMXBean;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;
import java.util.ArrayList;
import java.util.List;

/**
 * Samples the driver process (client JVM) to collect metrics such as CPU time,
 * GC time, heap usage, and direct memory.
 * Runs on a dedicated daemon thread and records snapshots periodically.
 */
public final class DriverProcess {

    /** Sampling interval in milliseconds. */
    private static final long PERIOD_MS = 200L;

    /** Synthetic process ID to unify driver metrics with TaskManager metrics. */
    static final String PROCESS_ID = "client";

    private final List<MemoryPoolMXBean> heapPools = new ArrayList<>();
    private final List<BufferPoolMXBean> directPools = new ArrayList<>();
    private final com.sun.management.OperatingSystemMXBean os;

    /** Per-tick samples: {timestampMs, tickHeapPeak, tickDirectBytes}. */
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
                    // Skip pools that do not support peak usage resets.
                }
            }
        }

        for (BufferPoolMXBean pool : ManagementFactory.getPlatformMXBeans(BufferPoolMXBean.class)) {
            if ("direct".equals(pool.getName())) {
                directPools.add(pool);
            }
        }
    }

    /** Starts metric sampling on a daemon thread. */
    public synchronized void start() {
        if (running) {
            return;
        }
        running = true;
        sample();

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

    /** Stops sampling and takes one final reading. */
    public synchronized void stop() {
        running = false;
        if (sampler != null) {
            sampler.interrupt();
            sampler = null;
        }
        sample();
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
        heapByteSeconds += heapUsed * PERIOD_MS / 1000L;

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

    /**
     * Returns the maximum heap peak recorded during the specified time window.
     */
    public synchronized long windowHeapPeak(long fromMs, long toMs) {
        long max = 0L;
        for (long[] s : series) {
            if (s[0] >= fromMs && s[0] <= toMs) {
                max = Math.max(max, s[1]);
            }
        }
        return max;
    }

    /**
     * Returns the maximum direct memory usage recorded during the specified time window.
     */
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
