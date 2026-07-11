package clustering.benchmark.metrics;

/** Engine-metrics collector. Spark fills this from a SparkListener; Flink has no
 *  direct equivalent, so engine counters come from the file written by Flink's
 *  built-in metric system ({@code clustering.metrics.FileMetricReporter}) and
 *  aggregated by {@link MetricsFile}. Fields with no Flink analogue (spill bytes,
 *  shuffle wait/write time, Spark "stages", failedTaskCount, executorRunTimeMs)
 *  are not collected — the driver leaves the corresponding RunResult fields null
 *  (N/A), distinct from a measured 0. */
public final class BenchmarkListener {

    /** Immutable snapshot of engine counters at end of run. */
    public static final class Snapshot {
        public long shuffleReadBytes;
        public long shuffleWriteBytes;
        public long inputBytes;
        public long outputBytes;
        public long jvmGcTimeMs;
        public long executorCpuTimeNs;
        public long taskCount;
        public long peakExecutorMemoryBytes;   // MAX single TM heap
        public long totalExecutorMemoryBytes;  // SUM of per-TM heap peaks
        // diskBytesSpilled, memoryBytesSpilled, shuffleFetchWaitTimeMs, shuffleWriteTimeNs,
        // failedTaskCount, stageCount, totalStageMs: no Flink analogue -> not collected (N/A).
    }

    /** Snapshot of the engine metrics, read from the file written by Flink's metric system
     *  ({@code clustering.metrics.FileMetricReporter}) and aggregated by {@link MetricsFile}.
     *  Fields with no Flink analogue (spill bytes, shuffle wait/write time, Spark "stages",
     *  failedTaskCount) stay 0. */
    public Snapshot snapshot() {
        return MetricsFile.read();
    }
}
