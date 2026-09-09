package clustering.benchmark.metrics;


import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Map;

/** One row of benchmark output, single-line JSON, conforming to
 *  contract/run_result.schema.json. Java mirror of the Spark {@code RunResult}.
 *
 *  Uses the neutral {@code engineConf} key. Only counters BOTH engines can measure on the
 *  same basis are kept here — see {@link BenchmarkListener} / {@link MetricsFile} for the
 *  cross-engine mapping. Everything Spark-only or definition-mismatched (shuffle bytes,
 *  spill, task/stage counts, driver-JVM metrics, unified-memory peaks) was dropped rather
 *  than emitted as an always-null placeholder. */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
    "runId", "framework", "profile", "startedAtIso", "finishedAtIso", "status", "errorMessage",
    "algorithm", "algorithmParams", "dataset", "datasetMetadata", "engineConf", "experimentMetadata",
    "nRows", "nPartitions", "nFeatures",
    "loadDurationMs", "fitDurationMs", "evalDurationMs", "totalDurationMs",
    "phaseDurationsMs", "phaseCpuTimeNs", "phaseGcTimeMs", "phaseMemoryGbHours",
    "phaseDriverCpuTimeNs", "phaseDriverGcTimeMs", "phaseDriverMemoryGbHours",
    "phaseWindowPeakExecutorMemoryBytes", "phaseWindowTotalExecutorMemoryBytes",
    "phaseWindowPeakDirectMemoryBytes", "phaseWindowTotalDirectMemoryBytes",
    "phaseWindowDriverPeakHeapBytes", "phaseWindowDriverPeakDirectMemoryBytes",
    "jvmGcTimeMs", "executorCpuTimeNs", "avgCpuCoresBusy", "cpuCoreHours",
    "peakExecutorMemoryBytes", "totalExecutorMemoryBytes", "memoryGbHours",
    "peakDirectMemoryBytes", "totalDirectMemoryBytes",
    "driverCpuTimeNs", "driverCpuCoreHours", "driverPeakHeapBytes", "driverGcTimeMs", "driverMemoryGbHours",
    "driverPeakDirectMemoryBytes",
    "nClusters", "noiseFraction", "silhouette",
    "silhouetteScoredPoints", "silhouetteSampleClusters", "silhouetteUnscoredPoints",
    "clusterSizes", "daviesBouldin", "calinskiHarabasz"
})
public class RunResult {

    // identity
    public String runId;
    public String framework;
    public String profile;
    public String startedAtIso;
    public String finishedAtIso;
    public String status;            // "ok" | "failed"
    public String errorMessage;      // null unless failed

    // experimental factors
    public String algorithm;
    public Map<String, String> algorithmParams;
    public String dataset;
    public Map<String, String> datasetMetadata;
    public Map<String, String> engineConf;
    public Map<String, String> experimentMetadata;

    // workload
    public long nRows;
    public int nPartitions;
    public Integer nFeatures;        // null when unknown

    // wall-clock timings (driver, ms)
    public long loadDurationMs;
    public long fitDurationMs;
    public long evalDurationMs;
    public long totalDurationMs;

    // Per-phase breakdown, keyed by phase name ("load" | "fit" | "eval"). Same key set in every
    // map, same names and semantics as the Spark port. `fit` is the interesting one: load is
    // dominated by I/O and eval by the silhouette sample, so the algorithm's own cost only
    // shows up there. Phases tile the run window (each boundary settle falls inside the phase
    // that just ended, idle, so it adds ~no CPU), hence the maps sum to the run-level totals.
    // Peaks are split per phase by a different mechanism — see the note below.
    public Map<String, Long> phaseDurationsMs;
    public Map<String, Long> phaseCpuTimeNs;
    public Map<String, Long> phaseGcTimeMs;
    public Map<String, Double> phaseMemoryGbHours;
    public Map<String, Long> phaseDriverCpuTimeNs;
    public Map<String, Long> phaseDriverGcTimeMs;
    public Map<String, Double> phaseDriverMemoryGbHours;

    // Per-phase peaks: the max over the sampling ticks that fell inside the phase. A peak,
    // unlike an accumulator, cannot be sliced out of two readings after the fact, so it is
    // bucketed as it arrives — from the reporter's time series (see MetricsFile.windowPeaks).
    // The heap figures are gap-free within the phase: the reporter resets the JVM's tracked
    // peak every tick, so a spike between two ticks still sets that tick's record. The direct
    // figures are genuinely sampled (no peak API for buffer pools), so a sub-tick spike can be
    // missed there and only peakDirectMemoryBytes may catch it. The one approximation heap
    // keeps is attribution: the tick straddling a boundary counts wholly towards one phase.
    public Map<String, Long> phaseWindowPeakExecutorMemoryBytes;
    public Map<String, Long> phaseWindowTotalExecutorMemoryBytes;
    public Map<String, Long> phaseWindowPeakDirectMemoryBytes;
    public Map<String, Long> phaseWindowTotalDirectMemoryBytes;
    public Map<String, Long> phaseWindowDriverPeakHeapBytes;
    public Map<String, Long> phaseWindowDriverPeakDirectMemoryBytes;

    // I/O volume is deliberately NOT reported. A source operator's numBytesOut counts
    // SERIALIZED bytes handed downstream, once per job — and this benchmark issues several
    // jobs per run, each re-reading the materialized copy — while Spark's
    // InputMetrics.bytesRead counts COMPRESSED bytes read off storage, once, because the
    // loaded frame is cached. Different unit AND different multiplicity (plus numBytesOut is
    // 0 for a chained source with no shuffle below it), so the two cannot share a column.
    // Removed on both engines together; see the Spark RunResult's matching note.

    // ENGINE counters — every one measured on the SAME basis on both engines (see
    // MetricsFile's class doc for the Spark<->Flink mapping).
    public long jvmGcTimeMs;
    public long executorCpuTimeNs;
    public Double avgCpuCoresBusy;         // executorCpuTimeNs / (totalDurationMs * 1e6) = avg cores busy; cross-comparable
    public double cpuCoreHours;            // executorCpuTimeNs / 3.6e12 = worker CPU burnt, in core-hours (grant accounting)
    public long peakExecutorMemoryBytes;   // MAX heap of single worker (TM)
    public long totalExecutorMemoryBytes;  // SUM of per-worker heap peaks (cluster footprint)
    public double memoryGbHours;           // time-integral of heap used across workers, in GB-hours —
                                            // total RAM actually consumed over the run, not a peak/sum snapshot
    public long peakDirectMemoryBytes;     // MAX off-heap NIO buffer memory of a single worker (TM)
    public long totalDirectMemoryBytes;    // SUM of per-worker direct-memory peaks (cluster off-heap footprint)

    // Driver JVM = the JobManager process under standalone Application Mode (standalone-job.sh),
    // where main() — and so all driver-local algorithm code — actually runs. Under session-mode
    // `flink run` the JobManager does no such work and these read ~0: not wrong, just not the
    // process where the computation happened. Spark analogue: SparkClusteringJob's own driver.
    public long driverCpuTimeNs;
    public double driverCpuCoreHours;
    public long driverPeakHeapBytes;
    public long driverGcTimeMs;
    public double driverMemoryGbHours;
    public long driverPeakDirectMemoryBytes;   // peak off-heap NIO buffer memory of the driver/JobManager

    // evaluation — each present only when its metric is named in EvaluationSpec.metrics;
    // otherwise null -> omitted from JSON (NON_NULL), matching the Spark RunResult.
    public Integer nClusters;        // null when 'nClusters' not requested
    public Double noiseFraction;     // null when 'noiseFraction' not requested
    public Double silhouette;        // null when 'silhouette' not computed

    // What the silhouette was computed ON. Present exactly when `silhouette` is, because the score
    // alone cannot be read safely: the achieved sample is Binomial around evaluation.sampleSize
    // rather than equal to it, and a sample that LOST whole clusters produces the same kind of
    // number as one that kept them all — except it reads HIGH, since `b` was then minimised over
    // the survivors. Compare silhouetteSampleClusters against nClusters.
    public Integer silhouetteScoredPoints;
    public Integer silhouetteSampleClusters;
    public Integer silhouetteUnscoredPoints;

    public Map<String, Long> clusterSizes;   // null when 'clusterSizes' not requested

    // Centroid-based internal indices, always on the FULL data (linear in n, so never sampled).
    // Opposite directions: daviesBouldin LOWER is better, calinskiHarabasz HIGHER is better and
    // unbounded, so CH only ranks labellings of the SAME dataset.
    public Double daviesBouldin;       // null when 'daviesBouldin' not requested
    public Double calinskiHarabasz;    // null when 'calinskiHarabasz' not requested

    /** Average number of CPU cores busy over the run: {@code executorCpuTimeNs /
     *  (totalDurationMs * 1e6)} (CPU-seconds per wall-second = effective CPU parallelism).
     *  Cross-framework comparable (same formula on Spark); divide by allocated cores for a
     *  0–1 utilization. null when duration is non-positive. */
    public static Double avgCpuCoresBusy(long cpuNs, long totalDurationMs) {
        if (totalDurationMs <= 0) {
            return null;
        }
        return cpuNs / (totalDurationMs * 1_000_000.0);
    }

    private static final double NANOS_PER_CORE_HOUR = 3.6e12;

    /** Worker CPU burnt over the run, in core-hours — the unit the PLGrid grant is billed in.
     *  Same formula on Spark. */
    public static double cpuCoreHours(long cpuNs) {
        return cpuNs / NANOS_PER_CORE_HOUR;
    }

    private static final double BYTE_SECONDS_PER_GB_HOUR = 1024.0 * 1024 * 1024 * 3600.0;

    /** Time-integral of heap used across workers, in GB-hours: total RAM actually CONSUMED
     *  over the run (Riemann sum of current heap-used samples), as opposed to
     *  {@code totalExecutorMemoryBytes} which is a sum of per-worker PEAKS. Same formula and
     *  sampling method on Spark (see {@code ProcessCpuPlugin}/{@code FileMetricReporter}). */
    public static double memoryGbHours(long heapByteSeconds) {
        return heapByteSeconds / BYTE_SECONDS_PER_GB_HOUR;
    }

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static String toJsonString(RunResult r) {
        try {
            return MAPPER.writeValueAsString(r);
        } catch (IOException e) {
            throw new RuntimeException("Failed to serialize RunResult", e);
        }
    }

    /** Atomic write to {@code <outputDir>/<runId>.json} (tmp + rename).
     *
     *  ATOMIC_MOVE with a REPLACE_EXISTING fallback, mirroring the Spark repo's writer: a
     *  plain REPLACE_EXISTING move is free to be implemented as delete-then-copy, which
     *  leaves a window where a result scraper reads a partial file. */
    public static Path writeToDir(RunResult r, String outputDir) throws IOException {
        Path dir = Paths.get(outputDir);
        Files.createDirectories(dir);
        Path target = dir.resolve(r.runId + ".json");
        Path tmp = dir.resolve("." + r.runId + ".json.tmp");
        Files.write(tmp, toJsonString(r).getBytes(StandardCharsets.UTF_8));
        try {
            Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
        }
        return target;
    }
}
