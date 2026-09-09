package clustering.metrics;

import clustering.benchmark.metrics.MetricsFile;
import org.apache.flink.metrics.Counter;
import org.apache.flink.metrics.Gauge;
import org.apache.flink.metrics.Metric;
import org.apache.flink.metrics.MetricConfig;
import org.apache.flink.metrics.MetricGroup;
import org.apache.flink.metrics.reporter.MetricReporter;
import org.apache.flink.metrics.reporter.Scheduled;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.io.Writer;
import java.lang.management.BufferPoolMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/** Plain-text Flink metric reporter for the benchmark: it writes the built-in Flink metrics
 *  we care about (record counts, JVM GC/CPU/heap) to a text file, so the driver
 *  ({@code BenchmarkRunner}) can read them back into the {@code RunResult} — WITHOUT touching
 *  any algorithm code (no per-operator accumulators) and WITHOUT the driver polling REST.
 *
 *  Why a custom file reporter (not Prometheus/Slf4j): the benchmark jobs are short BATCH
 *  jobs, so a PULL reporter (Prometheus, ~15s scrape) misses them; this flushes every
 *  {@link #report()} tick and on {@link #close()}, capturing even sub-second jobs. Output is
 *  plain text (no Jackson) so the reporter has zero deps beyond flink-metrics-core and can be
 *  dropped into {@code $FLINK_HOME/lib} on the cluster.
 *
 *  <h3>One file per reporter instance</h3>
 *  A MiniCluster (and a real cluster) runs a SEPARATE reporter instance per process
 *  (JobManager + each TaskManager). They would clobber a shared file, so each instance writes
 *  to {@code <path>.<uuid>}; {@link clustering.benchmark.metrics.MetricsFile} globs and
 *  aggregates all of them. On a cluster point {@code path} at the shared filesystem so every
 *  node's file is visible to the driver.
 *
 *  <h3>Lifetime &amp; why we track the max</h3>
 *  One standalone cluster runs exactly one RunConfig, so the files hold only that run. Both
 *  task-scoped counters (numRecordsIn, ...) and TaskManager JVM gauges (CPU/GC/heap) are torn
 *  down before the reporter's final {@link #close()} flush — tasks when their job ends, TM
 *  gauges when the cluster stops. A naive "write current live metrics" loses them, because the
 *  last flush sees an empty registry. So we record the MAX value ever observed per metric id
 *  (in every {@link #report()} tick and again on removal) in {@link #peak} and always write
 *  that. All kept metrics are either monotonic (numRecords counters, CPU.Time,
 *  GC.Time — max == final) or want-peak (Heap.Used — max == peak), so max is correct.
 *
 *  <h3>TruePeak: a gap-free peak alongside the sampled one</h3>
 *  {@code Status.JVM.Memory.Heap.Used}'s tracked max is only as good as the reporter interval —
 *  a spike shorter than one tick between two ticks is invisible to it, unlike Spark's
 *  {@code ProcessCpuPlugin}, which reads {@link MemoryPoolMXBean#getPeakUsage()} — a high-water
 *  mark the JVM itself tracks continuously, gap-free. This reporter reads that SAME API (summed
 *  over the heap pools) at every tick and writes it under a sibling
 *  {@code <id>.TruePeak} key, so {@code peakExecutorMemoryBytes}/{@code driverPeakHeapBytes} can
 *  use a peak that is comparable to Spark's on equal terms, not just equal in method-name.
 *
 *  <p>The peak is RESET at every tick (see {@link #takeHeapPeak}), not once in {@link #open}.
 *  Each reading is then the high-water mark of that tick alone, and since the ticks tile wall
 *  clock, max-across-ticks is still the run's true peak — while max across the ticks of one
 *  phase is that phase's true peak, which a single cumulative reading cannot give (differencing
 *  a record book yields "by how much the record was beaten", not the window's maximum). Spark's
 *  sampler does the identical read-and-reset, so both engines' peaks — run-level and per-phase —
 *  are one definition.
 *
 *  <h3>Direct memory: Netty's off-heap network/shuffle buffers</h3>
 *  Both engines allocate off-heap DIRECT NIO buffers for their network stacks (Flink's own
 *  shuffle, Spark's Netty-based one) regardless of any heap/off-heap execution-memory config —
 *  invisible to every heap-scoped field above. Read via {@link BufferPoolMXBean} (name
 *  {@code "direct"}), summed into a synthetic {@code Status.JVM.Memory.Direct.Used} sibling of
 *  the Heap.Used id at every tick. Unlike heap, buffer pools have NO JVM-tracked peak API (no
 *  {@code getPeakUsage} equivalent), so this is a max-of-1s-samples like heap was before
 *  TruePeak — a real, symmetric limitation shared with Spark's identical approach
 *  ({@code ProcessCpuPlugin.peakDirectUsed}), not a Flink-only gap.
 *
 *  <h3>A time series alongside the last-value file, for per-phase peaks</h3>
 *  The main file holds ONE value per metric — the last (or max) seen — which is all the
 *  run-level fields need. A per-phase peak cannot be derived from it: a peak is a record book,
 *  not an odometer, so differencing two readings of it yields "by how much the record was
 *  beaten", not the window's peak. And the reporter cannot bucket by phase itself, because it
 *  lives in the TaskManager process and has no channel to the JobManager that knows where the
 *  phases are.
 *
 *  So the per-tick readings are also appended, one line per tick, to a sibling
 *  {@code <path>.series} file as {@code <epochMillis>\t<id>\t<value>} — the tick's heap
 *  high-water mark under {@code .WindowPeak}, the current direct-memory reading under
 *  {@code .Current} (buffer pools have no peak to reset). The driver, which owns
 *  the phase boundaries, takes the max within each window ({@link
 *  clustering.benchmark.metrics.MetricsFile#windowPeaks}); for heap that max is gap-free within
 *  the phase, the only slack being the tick that straddles a boundary, which counts wholly
 *  towards one side. Appended rather than rewritten: the
 *  main file is rewritten whole every tick, so a growing series there would make the run's I/O
 *  quadratic. No temp+rename either — a reader tolerates a truncated final line, which is the
 *  only tear an append can produce.
 *
 *  Cost at 200 ms: two lines per tick, ~10 lines/s, so a few hundred kB per process over a
 *  ten-minute run. The Spark port needs no equivalent — its samples already travel to the
 *  driver, so it buckets them on arrival.
 *
 *  <h3>Heap.Used ALSO drives a byte-seconds integral</h3>
 *  Alongside the peak, every {@code Heap.Used} reading also feeds {@code heapSeconds}: a
 *  Riemann sum of (current value * elapsed seconds since the previous tick), written out as a
 *  synthetic {@code <id>.ByteSeconds} counter. That is the source of {@code memoryGbHours} in
 *  the RunResult — total RAM actually consumed over the run, not just its peak.
 *
 *  Config (flink-conf.yaml / config.yaml):
 *  <pre>
 *  metrics.reporter.file.factory.class: clustering.metrics.FileMetricReporterFactory
 *  metrics.reporter.file.path: /path/to/flink-metrics.txt
 *  metrics.reporter.file.interval: 200 MILLISECONDS
 *  </pre> */
public class FileMetricReporter implements MetricReporter, Scheduled {

    private static final Logger logger = LoggerFactory.getLogger(FileMetricReporter.class);

    /** Identifier fragments we keep (matched against the FULL metric identifier). */
    private static final String[] KEEP = {
        "numRecordsIn", "numRecordsOut",
        "GarbageCollector", "Status.JVM.CPU.Time", "Status.JVM.Memory.Heap.Used"
    };

    private Path path;
    /** Sibling append-only file carrying the per-tick current readings; see the class doc. */
    private Path seriesPath;
    private final Map<String, Metric> live = new ConcurrentHashMap<>();
    /** Max value ever observed per kept metric id — survives task/TM teardown. */
    private final Map<String, Long> peak = new ConcurrentHashMap<>();
    /** Cumulative heap-used time-integral per Heap.Used metric id: current value * elapsed
     *  seconds since the previous tick, summed across every {@link #report()} call — the
     *  Riemann sum behind {@code memoryGbHours} (total RAM CONSUMED over the run, as opposed
     *  to {@link #peak}'s single high-water mark). Survives teardown like {@link #peak}: the
     *  accumulator entry is never removed once created. */
    private final Map<String, Double> heapSeconds = new ConcurrentHashMap<>();
    /** Wall-clock time the integral starts from. Set in {@link #open}, NOT lazily on the
     *  first {@link #report()} tick: the benchmark issues several short Flink jobs per run
     *  (see {@code FlinkClusteringJob}'s class doc), each with its own fresh reporter, and a
     *  job can finish before the configured reporter interval ever fires a periodic tick — its
     *  ONLY {@code report()} call is then the final flush from {@link #close}. Starting the
     *  clock at {@code open} means that single call still integrates over the job's real
     *  elapsed time (a one-sample rectangle) instead of contributing zero. */
    private long lastTickNanos = -1L;
    /** Heap pools whose peak-since-reset the JVM tracks continuously; see the TruePeak doc
     *  above. Populated once in {@link #open}. */
    private List<MemoryPoolMXBean> heapPools = new ArrayList<>();
    /** The "direct" NIO buffer pool; see the class doc above. Populated once in {@link #open}. */
    private List<BufferPoolMXBean> directPools = new ArrayList<>();

    @Override
    public void open(MetricConfig config) {
        String base = config.getString("path", System.getProperty("clustering.metrics.file",
            System.getProperty("java.io.tmpdir") + "/flink-metrics.txt"));
        this.path = Paths.get(base + "." + UUID.randomUUID());
        this.seriesPath = Paths.get(this.path + ".series");
        this.lastTickNanos = System.nanoTime();
        for (MemoryPoolMXBean pool : ManagementFactory.getMemoryPoolMXBeans()) {
            if (pool.getType() == MemoryType.HEAP && pool.getPeakUsage() != null) {
                try {
                    pool.resetPeakUsage();
                    heapPools.add(pool);
                } catch (UnsupportedOperationException ignore) {
                    // pool doesn't support reset -> skip it rather than report a peak that
                    // silently includes usage from before this reporter/process started
                }
            }
        }
        for (BufferPoolMXBean pool : ManagementFactory.getPlatformMXBeans(BufferPoolMXBean.class)) {
            if ("direct".equals(pool.getName())) {
                directPools.add(pool);
            }
        }
    }

    @Override
    public void close() {
        report();  // final flush
    }

    @Override
    public void notifyOfAddedMetric(Metric metric, String metricName, MetricGroup group) {
        if (!(metric instanceof Counter) && !(metric instanceof Gauge)) {
            return;
        }
        String id = group.getMetricIdentifier(metricName);
        if (keep(id)) {
            live.put(id, metric);
        }
    }

    @Override
    public void notifyOfRemovedMetric(Metric metric, String metricName, MetricGroup group) {
        String id = group.getMetricIdentifier(metricName);
        if (!keep(id)) {
            return;
        }
        live.remove(id);
        // Preserve the final value of any torn-down metric (counter OR gauge) before it
        // disappears from the registry — tasks tear down per job, TM gauges at cluster stop.
        Long v = value(metric);
        if (v != null) {
            peak.merge(id, v, Math::max);
        }
    }

    @Override
    public synchronized void report() {
        if (path == null) {
            return;
        }
        long now = System.nanoTime();
        double dtSeconds = lastTickNanos < 0 ? 0.0 : (now - lastTickNanos) / 1e9;
        lastTickNanos = now;

        // Fold every currently-live metric into the peak map (max), then write the peak map —
        // so values survive the teardown that precedes the final close() flush. Heap.Used
        // metrics ALSO feed the byte-seconds integral: value * elapsed-seconds-since-last-tick,
        // using the CURRENT reading (not peak.merge's running max), so this really integrates
        // usage over time rather than re-summing the same high-water mark.
        for (Map.Entry<String, Metric> e : live.entrySet()) {
            Long v = value(e.getValue());
            if (v != null) {
                peak.merge(e.getKey(), v, Math::max);
                if (dtSeconds > 0 && e.getKey().contains("Status.JVM.Memory.Heap.Used")) {
                    heapSeconds.merge(e.getKey(), v * dtSeconds, Double::sum);
                }
            }
        }
        // Gap-free heap peak (see the TruePeak class doc): the JVM's own continuously-tracked
        // high-water mark, READ AND THEN RESET, so this tick's value covers this tick only. The
        // per-tick windows tile wall clock, so max-across-ticks is still the run's true peak —
        // and the same value, bucketed by the driver, is a true peak PER PHASE (see the series
        // section of the class doc). Merge/max, not overwrite: the reading is no longer
        // monotonic once it is reset every tick.
        long tickPeakHeap = takeHeapPeak();
        if (!heapPools.isEmpty()) {
            for (String id : new ArrayList<>(peak.keySet())) {
                if (id.endsWith("Status.JVM.Memory.Heap.Used")) {
                    peak.merge(id + ".TruePeak", tickPeakHeap, Math::max);
                }
            }
        }
        // Direct (off-heap NIO) memory: no continuous-peak API exists for buffer pools, so
        // merge/max across ticks here — same shape as heap's pre-TruePeak approach, and the
        // same limitation Spark's peakDirectUsed has. Derives its id from the Heap.Used id
        // (swap the metric-name suffix) purely to inherit its .taskmanager./.jobmanager. scope.
        if (!directPools.isEmpty()) {
            long directUsed = 0L;
            for (BufferPoolMXBean pool : directPools) {
                directUsed += pool.getMemoryUsed();
            }
            String heapSuffix = "Status.JVM.Memory.Heap.Used";
            for (String id : new ArrayList<>(peak.keySet())) {
                if (id.endsWith(heapSuffix)) {
                    String prefix = id.substring(0, id.length() - heapSuffix.length());
                    peak.merge(prefix + "Status.JVM.Memory.Direct.Used", directUsed, Math::max);
                }
            }
        }
        appendCurrentReadings(tickPeakHeap);
        logger.debug("PROBE report path={} live={} peak={}", path.getFileName(), live.size(), peak.size());
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Long> e : peak.entrySet()) {
            sb.append(e.getKey()).append('\t').append(e.getValue()).append('\n');
        }
        for (Map.Entry<String, Double> e : heapSeconds.entrySet()) {
            sb.append(e.getKey()).append(".ByteSeconds").append('\t').append(Math.round(e.getValue())).append('\n');
        }
        try {
            Path tmp = Paths.get(path.toString() + ".writing");
            try (Writer w = Files.newBufferedWriter(tmp, StandardCharsets.UTF_8)) {
                w.write(sb.toString());
            }
            Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException ignore) {
            // best-effort; a missed flush just leaves slightly stale metrics
        }
    }

    /** Append this tick's heap high-water mark and current direct reading to the series file.
     *
     *  The heap value is the tick's own peak (read-and-reset, see {@link #takeHeapPeak}), so a
     *  driver bucketing these by phase gets a gap-free peak per phase rather than a max of
     *  instantaneous samples; direct memory has no peak API and stays a sample. Read straight
     *  off the MXBeans rather than off Flink's gauges, so both engines sample the
     *  identical quantity from the identical API. The id prefix is borrowed from a live
     *  {@code Heap.Used} identifier purely to inherit its {@code .taskmanager.}/{@code
     *  .jobmanager.} scope and per-process uniqueness — the same trick the Direct.Used synthesis
     *  above uses. Before any such identifier exists (the very first ticks of a process) there
     *  is nothing to attribute a reading to, so the tick is skipped.
     *
     *  <p>A line is emitted under EVERY matching prefix, not the first one found. A MiniCluster
     *  runs JobManager and TaskManager in one JVM, so a single reporter instance sees both
     *  {@code .jobmanager.} and {@code .taskmanager.} scoped Heap.Used identifiers; picking one
     *  would attribute this JVM's reading to whichever scope {@code ConcurrentHashMap} iteration
     *  happened to yield first, leaving the other silently at zero. Same treatment the
     *  TruePeak and Direct.Used synthesis above already gives, and on a real cluster the two
     *  scopes live in separate processes with one prefix each, so nothing is double-counted. */
    private void appendCurrentReadings(long tickPeakHeap) {
        if (seriesPath == null || heapPools.isEmpty()) {
            return;
        }
        String heapSuffix = "Status.JVM.Memory.Heap.Used";
        List<String> prefixes = new ArrayList<>();
        for (String id : peak.keySet()) {
            if (id.endsWith(heapSuffix)) {
                prefixes.add(id.substring(0, id.length() - heapSuffix.length()));
            }
        }
        if (prefixes.isEmpty()) {
            return;
        }
        long directUsed = 0L;
        for (BufferPoolMXBean pool : directPools) {
            directUsed += pool.getMemoryUsed();
        }
        long ts = System.currentTimeMillis();
        StringBuilder line = new StringBuilder();
        for (String prefix : prefixes) {
            line.append(ts).append('\t').append(prefix).append(heapSuffix)
                .append(".WindowPeak\t").append(tickPeakHeap).append('\n');
            line.append(ts).append('\t').append(prefix)
                .append("Status.JVM.Memory.Direct.Used.Current\t").append(directUsed).append('\n');
        }
        try {
            Files.write(seriesPath, line.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException ignore) {
            // best-effort, exactly like the main flush: a lost tick costs one sample
        }
    }

    /** This tick's heap high-water mark: the JVM's continuously-tracked peak, read and then
     *  reset so the next tick starts a fresh window. Read and reset run pool by pool rather than
     *  in two passes, to keep the gap between them minimal. Every pool in {@link #heapPools}
     *  accepted a reset in {@link #open}, so none can refuse one here. */
    private long takeHeapPeak() {
        long total = 0L;
        for (MemoryPoolMXBean pool : heapPools) {
            total += pool.getPeakUsage().getUsed();
            pool.resetPeakUsage();
        }
        return total;
    }

    private static Long value(Metric m) {
        if (m instanceof Counter) {
            return ((Counter) m).getCount();
        }
        if (m instanceof Gauge) {
            Object v = ((Gauge<?>) m).getValue();
            if (v instanceof Number) {
                return ((Number) v).longValue();
            }
        }
        return null;
    }

    private static boolean keep(String id) {
        for (String k : KEEP) {
            if (id.contains(k)) {
                return true;
            }
        }
        return false;
    }
}
