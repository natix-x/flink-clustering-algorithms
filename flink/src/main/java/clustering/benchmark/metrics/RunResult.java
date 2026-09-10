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

/**
 * Represents a single row of benchmark output in JSON format.
 * Conforms to contract/run_result.schema.json.
 */
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

    // Identity
    public String runId;
    public String framework;
    public String profile;
    public String startedAtIso;
    public String finishedAtIso;
    public String status;
    public String errorMessage;

    // Experimental factors
    public String algorithm;
    public Map<String, String> algorithmParams;
    public String dataset;
    public Map<String, String> datasetMetadata;
    public Map<String, String> engineConf;
    public Map<String, String> experimentMetadata;

    // Workload
    public long nRows;
    public int nPartitions;
    public Integer nFeatures;

    // Wall-clock timings (ms)
    public long loadDurationMs;
    public long fitDurationMs;
    public long evalDurationMs;
    public long totalDurationMs;

    // Per-phase breakdown
    public Map<String, Long> phaseDurationsMs;
    public Map<String, Long> phaseCpuTimeNs;
    public Map<String, Long> phaseGcTimeMs;
    public Map<String, Double> phaseMemoryGbHours;
    public Map<String, Long> phaseDriverCpuTimeNs;
    public Map<String, Long> phaseDriverGcTimeMs;
    public Map<String, Double> phaseDriverMemoryGbHours;

    // Per-phase peaks
    public Map<String, Long> phaseWindowPeakExecutorMemoryBytes;
    public Map<String, Long> phaseWindowTotalExecutorMemoryBytes;
    public Map<String, Long> phaseWindowPeakDirectMemoryBytes;
    public Map<String, Long> phaseWindowTotalDirectMemoryBytes;
    public Map<String, Long> phaseWindowDriverPeakHeapBytes;
    public Map<String, Long> phaseWindowDriverPeakDirectMemoryBytes;

    // Engine counters (Worker)
    public long jvmGcTimeMs;
    public long executorCpuTimeNs;
    public Double avgCpuCoresBusy;
    public double cpuCoreHours;
    public long peakExecutorMemoryBytes;
    public long totalExecutorMemoryBytes;
    public double memoryGbHours;
    public long peakDirectMemoryBytes;
    public long totalDirectMemoryBytes;

    // Engine counters (Driver)
    public long driverCpuTimeNs;
    public double driverCpuCoreHours;
    public long driverPeakHeapBytes;
    public long driverGcTimeMs;
    public double driverMemoryGbHours;
    public long driverPeakDirectMemoryBytes;

    // Evaluation
    public Integer nClusters;
    public Double noiseFraction;
    public Double silhouette;

    public Integer silhouetteScoredPoints;
    public Integer silhouetteSampleClusters;
    public Integer silhouetteUnscoredPoints;

    public Map<String, Long> clusterSizes;

    // Centroid-based internal indices
    public Double daviesBouldin;
    public Double calinskiHarabasz;

    /**
     * Average number of CPU cores busy over the run.
     * Formula: {@code executorCpuTimeNs / (totalDurationMs * 1e6)}
     */
    public static Double avgCpuCoresBusy(long cpuNs, long totalDurationMs) {
        if (totalDurationMs <= 0) {
            return null;
        }
        return cpuNs / (totalDurationMs * 1_000_000.0);
    }

    private static final double NANOS_PER_CORE_HOUR = 3.6e12;

    /**
     * Worker CPU time burnt over the run, expressed in core-hours.
     */
    public static double cpuCoreHours(long cpuNs) {
        return cpuNs / NANOS_PER_CORE_HOUR;
    }

    private static final double BYTE_SECONDS_PER_GB_HOUR = 1024.0 * 1024 * 1024 * 3600.0;

    /**
     * Time-integral of heap used across workers, expressed in GB-hours.
     */
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

    /**
     * Atomically writes the RunResult to {@code <outputDir>/<runId>.json}
     * using a temporary file and a rename operation.
     */
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
