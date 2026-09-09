package clustering.benchmark.metrics;


import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/** Reads the plain-text files written by {@code clustering.metrics.FileMetricReporter} (one
 *  per JobManager/TaskManager process, named {@code <path>.<uuid>}) and aggregates Flink's
 *  built-in metrics into a {@link BenchmarkListener.Snapshot}.
 *
 *  Each line is {@code <metric-identifier>\t<long value>}. Counters are SUMMED across task
 *  instances; TaskManager JVM gauges are summed across TMs (each is that TM's run-end value).
 *  {@code .taskmanager.}-scoped JVM metrics feed the executor/worker fields (the TaskManager
 *  is the Flink analogue of a Spark executor); {@code .jobmanager.}-scoped JVM metrics feed the
 *  driver fields — but only meaningful under standalone Application Mode ({@code
 *  standalone-job.sh}), where the JobManager process is what actually runs the benchmark's
 *  driver-local algorithm code (see {@code BenchmarkRunner}'s class doc). Under session-mode
 *  {@code flink run} the JobManager does no such work and these fields read ~0 — not wrong,
 *  just not the process where the driver-local computation happened.
 *
 *  <h3>Cross-framework comparability (Spark port)</h3>
 *  To match the Spark {@code RunResult} on the SAME measurement basis. Note what is absent:
 *  I/O volume. A source's {@code numBytesOut} is SERIALIZED bytes handed downstream, counted
 *  once per job — and a run issues several, each re-reading the materialized copy — whereas
 *  Spark's {@code InputMetrics.bytesRead} is COMPRESSED bytes off storage, counted once
 *  because the frame is cached. Wrong unit and wrong multiplicity, so it was dropped from the
 *  contract on both engines rather than reported as if comparable.
 *  <ul>
 *    <li>{@code peakExecutorMemoryBytes} = JVM heap used, from the gap-free {@code TruePeak}
 *        (analogue of Spark {@code JVMHeapMemory}, itself {@code MemoryPoolMXBean
 *        #getPeakUsage()} — the same continuously-tracked API, not the periodic
 *        {@code Status.JVM.Memory.Heap.Used} gauge sample, which can miss a sub-second spike).
 *        Spark's off-heap has no clean Flink analogue (Flink managed memory ≠ Spark off-heap),
 *        so heap-only is the comparable basis.</li>
 *    <li>{@code heapByteSeconds} (-&gt; {@code memoryGbHours}) = Riemann-sum time-integral of
 *        heap used, written by the reporter itself (see {@code FileMetricReporter}) as a
 *        synthetic {@code <id>.ByteSeconds} counter, summed here like any other TM counter.
 *        Same sampling method as Spark's {@code ProcessCpuPlugin} accumulator.</li>
 *    <li>{@code peakDirectMemoryBytes}/{@code totalDirectMemoryBytes} = off-heap NIO buffer
 *        memory (Netty network/shuffle buffers on both engines), read via
 *        {@code BufferPoolMXBean} and synthesised as {@code Status.JVM.Memory.Direct.Used} by
 *        the reporter (see its class doc) — analogue of Spark's
 *        {@code ProcessCpuPlugin.peakDirectUsed}. Max-of-1s-samples on both sides (no
 *        JVM-tracked peak API for buffer pools), unlike heap's gap-free TruePeak.</li>
 *  </ul>
 *
 *  <h3>KNOWN LIMITATION: CPU/heap are overcounted on {@code local} profile (MiniCluster)</h3>
 *  {@code Status.JVM.CPU.Time} (and heap) is cumulative SINCE THE JVM PROCESS STARTED, never
 *  reset. On a real cluster that is exactly right: one TaskManager/JobManager process lives for
 *  the WHOLE run, so one reporter file per process holds one honest total, and summing across
 *  files sums across genuinely different processes. Under {@code local}, {@code EnvFactory}
 *  mints a fresh {@code createLocalEnvironment(...)} — a whole new MiniCluster, hence a new
 *  TaskExecutor/JobManager/{@code FileMetricReporter} instance with a fresh UUID file — for
 *  EVERY action (materialize, fit, each eval sub-job), all inside the SAME test/dev JVM. Each of
 *  those files' cumulative-since-start reading already includes every earlier phase's CPU, so
 *  summing N files inflates the total by roughly Nx (confirmed empirically: a 12-partition local
 *  kmeans run produced 5 reporter files reading 15.15/8.35/7.04/14.01/12.53 CPU-seconds, summing
 *  to the exact 57.07s this method reported as {@code executorCpuTimeNs} — the true figure is
 *  close to the single largest reading, not their sum). Deliberately left unfixed: it never
 *  reaches Ares production numbers (there, {@code EnvFactory} attaches to the ALREADY-running
 *  standalone cluster instead of creating one), so it costs nothing on the thesis's actual
 *  data — only local dev/test CPU and heap figures are meaningless as absolute numbers. A fix
 *  would mean this method taking MAX-across-files instead of SUM specifically for local-profile
 *  runs — worth doing only if local numbers become load-bearing for something.
 *
 *  <h3>TruePeak inherits the SUM half of this, not the MAX half</h3>
 *  Unlike {@code Status.JVM.CPU.Time}, the {@code .TruePeak} gauge (see {@code
 *  FileMetricReporter}) genuinely IS reset per reporter — {@code resetPeakUsage()} runs in
 *  {@code open()} and again at every tick — so each local-mode phase's file correctly holds only
 *  that phase's own peak,
 *  not a running total since JVM start. {@code peakExecutorMemoryBytes} (MAX-across-files) is
 *  therefore SAFE in local mode: taking the largest of several correctly-scoped per-phase peaks
 *  is exactly right. But {@code totalExecutorMemoryBytes} and {@code heapByteSeconds} (SUM-across-
 *  files, same as CPU) still add up N per-phase readings of what is, in local mode, the SAME
 *  underlying JVM heap sampled at N different (mostly non-overlapping) moments — inflating the
 *  "cluster footprint" total by roughly the number of local-mode phases, exactly like CPU. So:
 *  CPU (both MAX-per-worker and SUM-across-workers legs) and heap's SUM leg are equally affected;
 *  heap's MAX leg (peakExecutorMemoryBytes) is not.
 *
 *  A related, unconfirmed possibility was raised but not observed: if two local-mode reporters'
 *  lifetimes actually OVERLAPPED in time, the later one's {@code resetPeakUsage()} could zero out
 *  a heap pool the earlier one hadn't finished reading yet, UNDER-counting instead of over-. This
 *  cannot happen today — {@code FlinkClusteringJob}'s phases (and each multi-job algorithm's
 *  internal jobs) run strictly sequentially, so at most one local MiniCluster/reporter is ever
 *  alive at a time — but it is the same structural cause as the SUM bug above: a JVM-process-
 *  global resource ({@code MemoryPoolMXBean}, like {@code Status.JVM.CPU.Time}) sampled by
 *  multiple reporter INSTANCES that only make sense as one-per-process, which local mode alone
 *  violates by minting a fresh instance per job. Ares is immune to both for the identical reason:
 *  one TaskManager/JobManager process, one reporter, for the whole run — never two live at once,
 *  never a global resource shared across reporter instances. */
public final class MetricsFile {

    private MetricsFile() {}

    /** Base path the reporters write to (each appends {@code .<uuid>}): env
     *  {@code CLUSTERING_METRICS_FILE}, else system property {@code clustering.metrics.file},
     *  else a temp file. */
    public static String path() {
        String p = System.getenv("CLUSTERING_METRICS_FILE");
        if (p == null || p.isEmpty()) {
            p = System.getProperty("clustering.metrics.file",
                System.getProperty("java.io.tmpdir") + "/flink-metrics.txt");
        }
        return p;
    }

    /** True when the path was set explicitly (env or system property), i.e. NOT the
     *  node-local {@code java.io.tmpdir} fallback. On a real cluster the path must be an
     *  explicit shared-filesystem location, or the driver will only see its own node's
     *  files; this lets the caller fail fast in that case. */
    public static boolean isExplicitlyConfigured() {
        String env = System.getenv("CLUSTERING_METRICS_FILE");
        if (env != null && !env.isEmpty()) {
            return true;
        }
        return System.getProperty("clustering.metrics.file") != null;
    }

    /** Delete all reporter files for a clean run — the last-value files AND the appended time
     *  series. Both, or the series survives: `reporterFiles` deliberately excludes `.series`
     *  (different format), so listing only that would leave the previous run's samples on disk,
     *  where they would land in this run's phase windows and inflate every sampled peak. */
    public static void reset() {
        List<Path> stale = new ArrayList<>(reporterFiles());
        stale.addAll(seriesFiles());
        for (Path f : stale) {
            try {
                Files.deleteIfExists(f);
            } catch (IOException ignore) {
                // best-effort
            }
        }
    }

    /** All files written by the reporter instances: {@code <basename>.*} in the base dir. */
    private static List<Path> reporterFiles() {
        List<Path> out = new ArrayList<>();
        Path base = Paths.get(path());
        Path dir = base.getParent() == null ? Paths.get(".") : base.getParent();
        String prefix = base.getFileName().toString() + ".";
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, prefix + "*")) {
            for (Path p : ds) {
                String name = p.getFileName().toString();
                // `.series` holds the per-tick time series, a different format entirely
                // (see seriesFiles); `.writing` is a half-written main file.
                if (!name.endsWith(".writing") && !name.endsWith(".series")) {
                    out.add(p);
                }
            }
        } catch (IOException ignore) {
            // dir missing -> no files
        }
        return out;
    }

    /** Parse + aggregate all reporter files into a Snapshot. No files -> all zeros. */
    public static BenchmarkListener.Snapshot read() {
        BenchmarkListener.Snapshot s = new BenchmarkListener.Snapshot();
        long heap = 0, heapMax = 0;
        long driverHeapMax = 0;
        long direct = 0, directMax = 0, driverDirectMax = 0;

        for (Path file : reporterFiles()) {
            List<String> lines;
            try {
                lines = Files.readAllLines(file, StandardCharsets.UTF_8);
            } catch (IOException e) {
                continue;
            }
            for (String line : lines) {
                int tab = line.lastIndexOf('\t');
                if (tab < 0) {
                    continue;
                }
                String id = line.substring(0, tab);
                long v;
                try {
                    v = Long.parseLong(line.substring(tab + 1).trim());
                } catch (NumberFormatException e) {
                    continue;
                }
                boolean tm = id.contains(".taskmanager.");
                boolean jm = id.contains(".jobmanager.");
                if (tm && id.endsWith(".Status.JVM.CPU.Time")) {
                    s.cpuByProcess.merge(processOf(id), v, Math::max);
                } else if (tm && id.endsWith(".Status.JVM.Memory.Heap.Used.TruePeak")) {
                    // Gap-free JVM-tracked high-water mark (see FileMetricReporter's TruePeak
                    // doc) — comparable to Spark's MemoryPoolMXBean.getPeakUsage()-based peak on
                    // equal terms, unlike the plain Heap.Used gauge's sampled max below it.
                    heap += v;                      // SUM across TMs (cluster footprint)
                    heapMax = Math.max(heapMax, v); // MAX single TM
                } else if (tm && id.endsWith(".Status.JVM.Memory.Heap.Used.ByteSeconds")) {
                    s.heapSecondsByProcess.merge(processOf(id), v, Math::max);
                } else if (tm && id.endsWith(".Status.JVM.Memory.Direct.Used")) {
                    // No JVM-tracked peak API for buffer pools (see FileMetricReporter's
                    // direct-memory doc) — max of samples, same limitation as Spark's.
                    direct += v;                     // SUM across TMs (cluster off-heap footprint)
                    directMax = Math.max(directMax, v); // MAX single TM
                } else if (tm && id.contains("GarbageCollector") && id.endsWith(".Time")) {
                    // Young + Old land under one process key, hence sum rather than max.
                    s.gcByProcess.merge(processOf(id), v, Long::sum);
                } else if (jm && id.endsWith(".Status.JVM.CPU.Time")) {
                    s.driverCpuByProcess.merge(processOf(id), v, Math::max);
                } else if (jm && id.endsWith(".Status.JVM.Memory.Heap.Used.TruePeak")) {
                    driverHeapMax = Math.max(driverHeapMax, v);
                } else if (jm && id.endsWith(".Status.JVM.Memory.Heap.Used.ByteSeconds")) {
                    s.driverHeapSecondsByProcess.merge(processOf(id), v, Math::max);
                } else if (jm && id.endsWith(".Status.JVM.Memory.Direct.Used")) {
                    driverDirectMax = Math.max(driverDirectMax, v);
                } else if (jm && id.contains("GarbageCollector") && id.endsWith(".Time")) {
                    s.driverGcByProcess.merge(processOf(id), v, Long::sum);
                }
            }
        }

        // Peaks aggregate here and stay scalars: they are never differenced, so per-process
        // bookkeeping would buy nothing.
        s.peakExecutorMemoryBytes = heapMax;   // MAX single TM heap
        s.totalExecutorMemoryBytes = heap;     // SUM of per-TM heap peaks (cluster footprint)
        s.peakDirectMemoryBytes = directMax;   // MAX single TM direct memory
        s.totalDirectMemoryBytes = direct;     // SUM of per-TM direct-memory peaks
        s.driverPeakHeapBytes = driverHeapMax;
        s.driverPeakDirectMemoryBytes = driverDirectMax;
        // Convenience totals over the per-process maps, for callers that want the raw sum.
        s.recomputeTotals();
        return s;
    }

    /** The process a metric identifier belongs to: everything before {@code Status.JVM.}, which
     *  is {@code <host>.taskmanager.<tm-id>.} or {@code <host>.jobmanager.} — unique per JVM and
     *  stable across the run.
     *
     *  <p>This key is what makes a cumulative counter differenceable. Summing across processes
     *  FIRST and subtracting the sums second is only correct while the set of processes is
     *  identical at both reads; a TaskManager whose file is missing at one read and present at
     *  another silently breaks the telescoping, and the error shows up as phases that do not add
     *  up to the run (observed on a 4-node run, 29.08.2026: worker CPU phases summed to 44.8 s
     *  against a run total of 31.7 s, while the single-file JobManager legs matched exactly —
     *  which is precisely the fingerprint of a membership change). Subtracting per process and
     *  summing after is what the Spark port has always done ({@code
     *  ProcessCpuPlugin.sumDelta}); this brings Flink onto the same arithmetic. */
    private static String processOf(String id) {
        int marker = id.indexOf("Status.JVM.");
        return marker < 0 ? id : id.substring(0, marker);
    }

    /** The append-only time-series files, one per reporter instance. */
    private static List<Path> seriesFiles() {
        List<Path> out = new ArrayList<>();
        Path base = Paths.get(path());
        Path dir = base.getParent() == null ? Paths.get(".") : base.getParent();
        String prefix = base.getFileName().toString() + ".";
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, prefix + "*.series")) {
            for (Path p : ds) {
                out.add(p);
            }
        } catch (IOException ignore) {
            // dir missing -> no files
        }
        return out;
    }

    /** Heap/direct peaks within one wall-clock window, from the reporter time series.
     *
     *  <p>Why the series and not a difference of two readings of the run-level peak: that peak
     *  is a record book, so differencing two readings of it gives "by how much the record was
     *  beaten in this window", not the window's peak — the information is simply not in two
     *  readings. Bucketing per-tick readings is the only way to get a per-phase figure without a
     *  driver -> worker control channel, which Flink's reporter does not have (it lives in the
     *  TaskManager process). The Spark port does exactly the same thing, just with the samples
     *  arriving over RPC instead of through a file, so the two engines' numbers are one
     *  definition.
     *
     *  <p>The heap series carries each tick's own high-water mark (the reporter resets the JVM
     *  peak every tick, see {@code FileMetricReporter#takeHeapPeak}), so the max taken here is
     *  gap-free within the window: a spike between two ticks still set the record inside its own
     *  tick. Direct memory has no peak API on the JDK's buffer pools, so its series is genuinely
     *  sampled and a sub-tick spike can be missed — the one asymmetry left, and identical on
     *  Spark.
     *
     *  <p>Per-process granularity comes free: a series id carries the process's own metric
     *  scope, so max-across-ids is "heaviest single worker" and sum-across-ids is "cluster
     *  footprint" — mirroring the run-level peak/total pair.
     *
     *  <p>One caveat this has and nothing else here does: the timestamps come from each
     *  TaskManager's own clock while the window bounds come from the JobManager's. NTP on Ares
     *  keeps that within milliseconds, i.e. far inside a 200 ms bucket, but a badly skewed node
     *  would misattribute samples near a boundary. Spark has no equivalent risk — its driver
     *  stamps arrival itself. */
    public static BenchmarkListener.PhasePeaks windowPeaks(long fromMs, long toMs) {
        // id -> max current reading inside the window
        java.util.Map<String, Long> maxById = new java.util.HashMap<>();
        for (Path file : seriesFiles()) {
            List<String> lines;
            try {
                lines = Files.readAllLines(file, StandardCharsets.UTF_8);
            } catch (IOException e) {
                continue;
            }
            for (String line : lines) {
                int firstTab = line.indexOf('\t');
                int lastTab = line.lastIndexOf('\t');
                if (firstTab < 0 || lastTab <= firstTab) {
                    continue;   // truncated final line of an append — expected, skip it
                }
                long ts;
                long v;
                try {
                    ts = Long.parseLong(line.substring(0, firstTab).trim());
                    v = Long.parseLong(line.substring(lastTab + 1).trim());
                } catch (NumberFormatException e) {
                    continue;
                }
                if (ts < fromMs || ts > toMs) {
                    continue;
                }
                String id = line.substring(firstTab + 1, lastTab);
                maxById.merge(id, v, Math::max);
            }
        }

        BenchmarkListener.PhasePeaks p = new BenchmarkListener.PhasePeaks();
        for (java.util.Map.Entry<String, Long> e : maxById.entrySet()) {
            String id = e.getKey();
            long v = e.getValue();
            boolean tm = id.contains(".taskmanager.");
            boolean jm = id.contains(".jobmanager.");
            boolean heap = id.endsWith("Status.JVM.Memory.Heap.Used.WindowPeak");
            boolean direct = id.endsWith("Status.JVM.Memory.Direct.Used.Current");
            if (tm && heap) {
                p.maxWorkerHeapBytes = Math.max(p.maxWorkerHeapBytes, v);
                p.totalWorkerHeapBytes += v;
            } else if (tm && direct) {
                p.maxWorkerDirectBytes = Math.max(p.maxWorkerDirectBytes, v);
                p.totalWorkerDirectBytes += v;
            } else if (jm && heap) {
                p.driverHeapBytes = Math.max(p.driverHeapBytes, v);
            } else if (jm && direct) {
                p.driverDirectBytes = Math.max(p.driverDirectBytes, v);
            }
        }
        return p;
    }
}
