package clustering.benchmark.metrics;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Map;

/** One row of benchmark output, single-line JSON, conforming to
 *  contract/run_result.schema.json. Java mirror of the Spark {@code RunResult}.
 *
 *  Uses the neutral {@code engineConf} key. ENGINE-tagged counters
 *  (shuffle/spill/stage/executor) are populated where Flink has an analogue and
 *  left 0 otherwise (see {@link BenchmarkListener}). */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
    "runId", "framework", "profile", "startedAtIso", "finishedAtIso", "status", "errorMessage",
    "algorithm", "algorithmParams", "dataset", "datasetMetadata", "engineConf", "experimentMetadata",
    "nRows", "nPartitions", "nFeatures",
    "loadDurationMs", "fitDurationMs", "evalDurationMs", "totalDurationMs",
    "shuffleReadBytes", "shuffleWriteBytes", "inputBytes", "outputBytes",
    "diskBytesSpilled", "memoryBytesSpilled", "jvmGcTimeMs", "executorCpuTimeNs", "executorRunTimeMs",
    "avgCpuCoresBusy", "shuffleFetchWaitTimeMs", "shuffleWriteTimeNs",
    "taskCount", "failedTaskCount", "stageCount", "totalStageMs",
    "peakExecutorMemoryBytes", "totalExecutorMemoryBytes",
    "nClusters", "noiseFraction", "silhouette", "clusterSizes"
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

    // ENGINE counters. Fields Flink CANNOT measure (no architectural analogue) are
    // Long and left null -> omitted from JSON (NON_NULL) -> NaN/"N/A" in analysis,
    // which is distinct from a measured 0 (e.g. Spark diskBytesSpilled=0 = no spill).
    public long shuffleReadBytes;
    public long shuffleWriteBytes;         // proxy: = shuffleReadBytes (Flink has no write split)
    public long inputBytes;
    public long outputBytes;
    public Long diskBytesSpilled;          // N/A in Flink
    public Long memoryBytesSpilled;        // N/A in Flink
    public long jvmGcTimeMs;
    public long executorCpuTimeNs;
    public Long executorRunTimeMs;         // N/A in Flink (Spark = per-task sum; not comparable)
    public Double avgCpuCoresBusy;         // executorCpuTimeNs / (totalDurationMs * 1e6) = avg cores busy; cross-comparable
    public Long shuffleFetchWaitTimeMs;    // N/A in Flink
    public Long shuffleWriteTimeNs;        // N/A in Flink
    public long taskCount;                 // framework-internal definition (not Spark-comparable)
    public Long failedTaskCount;           // N/A in Flink
    public Long stageCount;                // N/A in Flink (no stages)
    public Long totalStageMs;              // N/A in Flink (no stages)
    public long peakExecutorMemoryBytes;   // MAX heap of single worker (TM)
    public long totalExecutorMemoryBytes;  // SUM of per-worker heap peaks (cluster footprint)

    // evaluation — each present only when its metric is named in EvaluationSpec.metrics;
    // otherwise null -> omitted from JSON (NON_NULL), matching the Spark RunResult.
    public Integer nClusters;        // null when 'nClusters' not requested
    public Double noiseFraction;     // null when 'noiseFraction' not requested
    public Double silhouette;        // null when 'silhouette' not computed
    public Map<String, Long> clusterSizes;   // null when 'clusterSizes' not requested

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

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static String toJsonString(RunResult r) {
        try {
            return MAPPER.writeValueAsString(r);
        } catch (IOException e) {
            throw new RuntimeException("Failed to serialize RunResult", e);
        }
    }

    /** Atomic-ish write to {@code <outputDir>/<runId>.json} (tmp + rename). */
    public static Path writeToDir(RunResult r, String outputDir) throws IOException {
        Path dir = Paths.get(outputDir);
        Files.createDirectories(dir);
        Path target = dir.resolve(r.runId + ".json");
        Path tmp = dir.resolve("." + r.runId + ".json.tmp");
        Files.write(tmp, toJsonString(r).getBytes(StandardCharsets.UTF_8));
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
        return target;
    }
}
