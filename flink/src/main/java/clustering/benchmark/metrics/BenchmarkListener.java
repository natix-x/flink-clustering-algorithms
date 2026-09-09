package clustering.benchmark.metrics;


/** Engine-metrics collector. Spark fills this from a SparkListener; Flink has no
 *  direct equivalent, so engine counters come from the file written by Flink's
 *  built-in metric system ({@code clustering.metrics.FileMetricReporter}) and
 *  aggregated by {@link MetricsFile}. Only counters measured on the same basis as
 *  the Spark side are collected — see {@link MetricsFile} for the mapping. */
public final class BenchmarkListener {

    /** Snapshot of engine counters at one instant.
     *
     *  <p>Cumulative counters are held PER PROCESS, not pre-summed. Summing first and
     *  subtracting the sums second is only correct while the set of processes is identical at
     *  both reads — see {@link MetricsFile#processOf} for the 4-node run where it was not. The
     *  scalar totals below are conveniences recomputed from the maps; every subtraction goes
     *  through the maps. Peaks are scalars for real: they are never differenced. */
    public static final class Snapshot {
        public final java.util.Map<String, Long> cpuByProcess = new java.util.HashMap<>();
        public final java.util.Map<String, Long> gcByProcess = new java.util.HashMap<>();
        public final java.util.Map<String, Long> heapSecondsByProcess = new java.util.HashMap<>();
        public final java.util.Map<String, Long> driverCpuByProcess = new java.util.HashMap<>();
        public final java.util.Map<String, Long> driverGcByProcess = new java.util.HashMap<>();
        public final java.util.Map<String, Long> driverHeapSecondsByProcess = new java.util.HashMap<>();

        public long jvmGcTimeMs;
        public long executorCpuTimeNs;
        public long peakExecutorMemoryBytes;   // MAX single TM heap
        public long totalExecutorMemoryBytes;  // SUM of per-TM heap peaks
        public long heapByteSeconds;           // time-integral of heap used, SUM across TMs
        public long peakDirectMemoryBytes;     // MAX single TM direct (off-heap NIO) buffer memory
        public long totalDirectMemoryBytes;    // SUM of per-TM direct-memory peaks
        public long driverCpuTimeNs;           // JobManager process CPU (Application Mode only)
        public long driverPeakHeapBytes;       // JobManager peak heap used
        public long driverHeapByteSeconds;     // JobManager heap-used time-integral
        public long driverGcTimeMs;            // JobManager whole-JVM GC time
        public long driverPeakDirectMemoryBytes; // JobManager peak direct (off-heap NIO) buffer memory

        /** Refresh the scalar totals from the per-process maps. */
        public void recomputeTotals() {
            executorCpuTimeNs = sum(cpuByProcess);
            jvmGcTimeMs = sum(gcByProcess);
            heapByteSeconds = sum(heapSecondsByProcess);
            driverCpuTimeNs = sum(driverCpuByProcess);
            driverGcTimeMs = sum(driverGcByProcess);
            driverHeapByteSeconds = sum(driverHeapSecondsByProcess);
        }

        private static long sum(java.util.Map<String, Long> m) {
            long total = 0L;
            for (long v : m.values()) {
                total += v;
            }
            return total;
        }
    }

    /** Values already accrued when the run window opened; null until {@link #captureBaseline}. */
    private Snapshot baseline;

    /** The DRIVER's own counters. Under session mode `main()` runs in the client JVM, which is not
     *  a Flink process and so has no reporter writing to the metrics file — the JobManager's
     *  `.jobmanager.` lines describe a process that only coordinates. See {@link DriverProcess}
     *  for why the bootstrap changed and why this is closer to Spark, not further from it. */
    private final DriverProcess driver = new DriverProcess();

    /** Begin sampling the driver process. Call once, next to {@link #captureBaseline}. */
    public void startDriverSampling() {
        driver.start();
    }

    /** Stop sampling; takes one final reading so the tail of the run is not lost. */
    public void stopDriverSampling() {
        driver.stop();
    }

    /** Overlays the driver legs of a file-derived snapshot with THIS process's readings, keyed
     *  under one synthetic process id so every per-process delta below applies unchanged. */
    private Snapshot withDriver(Snapshot s) {
        s.driverCpuByProcess.clear();
        s.driverGcByProcess.clear();
        s.driverHeapSecondsByProcess.clear();
        s.driverCpuByProcess.put(DriverProcess.PROCESS_ID, driver.cpuNanos());
        s.driverGcByProcess.put(DriverProcess.PROCESS_ID, driver.gcTimeMs());
        s.driverHeapSecondsByProcess.put(DriverProcess.PROCESS_ID, driver.heapByteSeconds());
        s.driverPeakHeapBytes = driver.peakHeapBytes();
        s.driverPeakDirectMemoryBytes = driver.peakDirectBytes();
        return s;
    }

    /** Freeze the current counter values as the run's zero point.
     *
     *  <p>Flink's {@code Status.JVM.CPU.Time} and {@code GarbageCollector.*.Time} are cumulative
     *  SINCE THE JVM PROCESS STARTED and are never reset, whereas Spark's {@code ProcessCpuPlugin}
     *  publishes a DELTA from plugin init. That difference is not cosmetic: TaskManagers are
     *  launched before the JobManager boots and then idle through slot registration, so a raw
     *  Flink reading charges every run for per-TM startup that the Spark reading excludes — and
     *  the error grows linearly with node count, i.e. it bends the CPU-efficiency-vs-nodes curve
     *  the benchmark exists to plot. Calling this once the cluster is ready, just before the
     *  timing window opens, puts both engines on the same basis.
     *
     *  <p>Only genuinely cumulative fields are rebased. Peaks are maxima over the process
     *  lifetime, not sums, so subtracting a baseline from one would be meaningless. */
    public void captureBaseline() {
        baseline = withDriver(MetricsFile.read());
    }

    /** Snapshot of the engine metrics, read from the file written by Flink's metric system
     *  ({@code clustering.metrics.FileMetricReporter}) and aggregated by {@link MetricsFile}. */
    public Snapshot snapshot() {
        Snapshot s = withDriver(MetricsFile.read());
        if (baseline == null) {
            return s;
        }
        // Rebase through the SAME per-process arithmetic the phases use, so the run window and
        // the phases inside it can no longer disagree about the same counter.
        PhaseDelta d = between(baseline, s);
        s.executorCpuTimeNs = d.cpuNanos;
        s.jvmGcTimeMs = d.gcTimeMs;
        s.heapByteSeconds = d.heapByteSeconds;
        s.driverCpuTimeNs = d.driverCpuNanos;
        s.driverGcTimeMs = d.driverGcTimeMs;
        s.driverHeapByteSeconds = d.driverHeapByteSeconds;
        return s;
    }

    /** Sum over processes of each one's counter delta, floored at 0 PER PROCESS.
     *
     *  <p>Iterates the later reading's processes and looks the earlier one up, defaulting to 0:
     *  a TaskManager that appeared after `from` contributes its whole reading, one that vanished
     *  contributes nothing, and a restarted one cannot contribute a negative. Identical rule and
     *  identical shape to the Spark port's {@code ProcessCpuPlugin.sumDelta} — which is the
     *  point, since the two numbers end up in the same column. */
    private static long sumDelta(java.util.Map<String, Long> to, java.util.Map<String, Long> from) {
        long total = 0L;
        for (java.util.Map.Entry<String, Long> e : to.entrySet()) {
            total += Math.max(0L, e.getValue() - from.getOrDefault(e.getKey(), 0L));
        }
        return total;
    }

    // ── Per-phase windows ─────────────────────────────────────────────────────
    // The baseline above is one instance of a general move: subtract two readings of a
    // cumulative counter. Taking a reading at each phase boundary instead of only at the run's
    // start gives per-phase resource use — which is what the thesis actually wants, since load
    // is dominated by I/O and eval by the silhouette sample, so the algorithm's own cost is
    // visible only in `fit`.
    //
    // What this measures is CPU/GC/heap burnt by the worker JVMs WHILE THE DRIVER WAS INSIDE
    // the phase — not use attributable to the phase's work, which a process-level counter
    // cannot express (GC catching up on the previous phase's garbage lands in the next phase).
    // The Spark port defines it identically (see {@code ProcessCpuPlugin.snapshot}), so the
    // cross-engine comparison stays honest; only the absolute attribution is approximate.
    //
    // Peaks take a different route to the same per-phase split. They cannot be differenced (a
    // record book is not an odometer), and the reporter cannot bucket by phase itself — it runs
    // in the TaskManager process and has no channel to the JobManager that knows where the
    // boundaries are. So the reporter resets {@code resetPeakUsage} every TICK instead of at a
    // boundary: each series line is that tick's own high-water mark, the ticks tile wall clock,
    // and the driver takes the max over the ticks inside a window. The run-level peak survives
    // (it is the max over every tick) and each phase gets a gap-free one. Spark's plugin does
    // the identical read-and-reset, so the two engines' peaks are one definition.

    /** Resource use between two marks: one phase, or the whole run window.
     *
     *  The driver/JobManager legs sit alongside the worker ones rather than folded in: a
     *  driver-local phase (PAM's swap loop, CLARA's seeding, dbscanpp's union-find) is invisible
     *  to every TaskManager-side counter, and locally the two are the SAME process, so they must
     *  never be summed. */
    public static final class PhaseDelta {
        public long cpuNanos;
        public long gcTimeMs;
        public long heapByteSeconds;
        public long driverCpuNanos;
        public long driverGcTimeMs;
        public long driverHeapByteSeconds;
    }

    /** Heap/direct peaks of one phase. Mirrors the run-level peak fields one-for-one, and the
     *  Spark port's {@code ProcessCpuPlugin.PhasePeaks} field for field. Heap is gap-free within
     *  the phase; direct memory is a max of samples, the JDK offering no peak API for buffer
     *  pools. */
    public static final class PhasePeaks {
        public long maxWorkerHeapBytes;
        public long totalWorkerHeapBytes;
        public long maxWorkerDirectBytes;
        public long totalWorkerDirectBytes;
        public long driverHeapBytes;
        public long driverDirectBytes;
    }

    /** Peaks within one wall-clock window — see {@link MetricsFile#windowPeaks}, which
     *  carries the reasoning and the one caveat (TaskManager clocks vs the JobManager's). */
    public PhasePeaks windowPeaks(long fromMs, long toMs) {
        PhasePeaks p = MetricsFile.windowPeaks(fromMs, toMs);
        // Worker legs from the reporter files, driver legs from this process — same window, same
        // read-and-reset rule, so the two are one definition rather than two.
        p.driverHeapBytes = driver.windowHeapPeak(fromMs, toMs);
        p.driverDirectBytes = driver.windowDirectPeak(fromMs, toMs);
        return p;
    }

    /** Read every cumulative counter as it stands now, WITHOUT rebasing — the raw zero point a
     *  phase is measured from. Unlike {@link #snapshot()} this is not the run window; it is one
     *  instant.
     *
     *  <p>Reading means re-parsing the reporter files, so on a real cluster the caller must let
     *  one reporter interval pass first (see {@code FlinkClusteringJob}'s settle) or the mark
     *  lands before the last flush. Locally each phase's MiniCluster flushes on {@code close()},
     *  so the boundary is already sharp. */
    public Snapshot mark() {
        return withDriver(MetricsFile.read());
    }

    /** Counter deltas between two marks.
     *
     *  <p>Floored at 0, and note WHERE the floor applies: {@link MetricsFile} already sums
     *  across TaskManagers, so unlike the Spark side (which holds per-executor maps and floors
     *  each one) this floors the sum. Identical to what {@link #since} has always done for the
     *  run window. */
    public static PhaseDelta between(Snapshot from, Snapshot to) {
        PhaseDelta d = new PhaseDelta();
        d.cpuNanos = sumDelta(to.cpuByProcess, from.cpuByProcess);
        d.gcTimeMs = sumDelta(to.gcByProcess, from.gcByProcess);
        d.heapByteSeconds = sumDelta(to.heapSecondsByProcess, from.heapSecondsByProcess);
        d.driverCpuNanos = sumDelta(to.driverCpuByProcess, from.driverCpuByProcess);
        d.driverGcTimeMs = sumDelta(to.driverGcByProcess, from.driverGcByProcess);
        d.driverHeapByteSeconds = sumDelta(to.driverHeapSecondsByProcess, from.driverHeapSecondsByProcess);
        return d;
    }
}
