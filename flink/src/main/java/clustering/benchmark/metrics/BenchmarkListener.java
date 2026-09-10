package clustering.benchmark.metrics;

/**
 * Engine-metrics collector for Flink.
 * Aggregates counters from Flink's built-in metric system and local driver metrics.
 */
public final class BenchmarkListener {

    /**
     * Snapshot of engine counters at a single point in time.
     * Contains cumulative counters per process and scalar totals.
     */
    public static final class Snapshot {
        public final java.util.Map<String, Long> cpuByProcess = new java.util.HashMap<>();
        public final java.util.Map<String, Long> gcByProcess = new java.util.HashMap<>();
        public final java.util.Map<String, Long> heapSecondsByProcess = new java.util.HashMap<>();
        public final java.util.Map<String, Long> driverCpuByProcess = new java.util.HashMap<>();
        public final java.util.Map<String, Long> driverGcByProcess = new java.util.HashMap<>();
        public final java.util.Map<String, Long> driverHeapSecondsByProcess = new java.util.HashMap<>();

        public long jvmGcTimeMs;
        public long executorCpuTimeNs;
        public long peakExecutorMemoryBytes;
        public long totalExecutorMemoryBytes;
        public long heapByteSeconds;
        public long peakDirectMemoryBytes;
        public long totalDirectMemoryBytes;
        public long driverCpuTimeNs;
        public long driverPeakHeapBytes;
        public long driverHeapByteSeconds;
        public long driverGcTimeMs;
        public long driverPeakDirectMemoryBytes;

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

    private Snapshot baseline;
    private final DriverProcess driver = new DriverProcess();

    public void startDriverSampling() {
        driver.start();
    }

    public void stopDriverSampling() {
        driver.stop();
    }

    /**
     * Overlays driver metrics onto a file-derived snapshot under a synthetic process ID.
     */
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

    /**
     * Freezes the current cumulative counter values as the baseline for the run.
     */
    public void captureBaseline() {
        baseline = withDriver(MetricsFile.read());
    }

    /**
     * Returns a snapshot of the current engine metrics, rebased against the baseline if captured.
     */
    public Snapshot snapshot() {
        Snapshot s = withDriver(MetricsFile.read());
        if (baseline == null) {
            return s;
        }

        PhaseDelta d = between(baseline, s);
        s.executorCpuTimeNs = d.cpuNanos;
        s.jvmGcTimeMs = d.gcTimeMs;
        s.heapByteSeconds = d.heapByteSeconds;
        s.driverCpuTimeNs = d.driverCpuNanos;
        s.driverGcTimeMs = d.driverGcTimeMs;
        s.driverHeapByteSeconds = d.driverHeapByteSeconds;
        return s;
    }

    /**
     * Computes the per-process delta between two maps, floored at zero per process.
     */
    private static long sumDelta(java.util.Map<String, Long> to, java.util.Map<String, Long> from) {
        long total = 0L;
        for (java.util.Map.Entry<String, Long> e : to.entrySet()) {
            total += Math.max(0L, e.getValue() - from.getOrDefault(e.getKey(), 0L));
        }
        return total;
    }

    /**
     * Represents cumulative resource usage differences between two marks.
     */
    public static final class PhaseDelta {
        public long cpuNanos;
        public long gcTimeMs;
        public long heapByteSeconds;
        public long driverCpuNanos;
        public long driverGcTimeMs;
        public long driverHeapByteSeconds;
    }

    /**
     * Tracks peak heap and direct memory within a specific phase.
     */
    public static final class PhasePeaks {
        public long maxWorkerHeapBytes;
        public long totalWorkerHeapBytes;
        public long maxWorkerDirectBytes;
        public long totalWorkerDirectBytes;
        public long driverHeapBytes;
        public long driverDirectBytes;
    }

    /**
     * Retrieves memory peaks that occurred within a specific wall-clock time window.
     */
    public PhasePeaks windowPeaks(long fromMs, long toMs) {
        PhasePeaks p = MetricsFile.windowPeaks(fromMs, toMs);
        p.driverHeapBytes = driver.windowHeapPeak(fromMs, toMs);
        p.driverDirectBytes = driver.windowDirectPeak(fromMs, toMs);
        return p;
    }

    /**
     * Reads all cumulative counters in their raw state without rebasing.
     */
    public Snapshot mark() {
        return withDriver(MetricsFile.read());
    }

    /**
     * Computes the differences of cumulative metrics between two snapshots.
     */
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
