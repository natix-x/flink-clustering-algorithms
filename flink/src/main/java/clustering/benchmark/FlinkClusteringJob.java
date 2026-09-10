package clustering.benchmark;

import clustering.benchmark.metrics.MetricsFile;
import clustering.metrics.FileMetricReporter;
import clustering.benchmark.config.ClusterProfile;
import clustering.benchmark.config.Params;
import clustering.benchmark.config.RunConfig;
import clustering.benchmark.datasource.DataSource;
import clustering.benchmark.evaluation.EvaluationResult;
import clustering.benchmark.evaluation.EvaluationRunner;
import clustering.benchmark.metrics.BenchmarkListener;
import clustering.benchmark.metrics.RunResult;
import clustering.benchmark.registry.AlgorithmRegistry;
import clustering.benchmark.registry.DataSourceRegistry;
import clustering.core.Clusterer;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.api.common.RuntimeExecutionMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Instant;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Executes a single clustering benchmark run on Flink.
 * The pipeline consists of three distributed phases:
 * 1. Load (distributed count)
 * 2. Fit (algorithm-specific execution)
 * 3. Evaluate (distributed cluster sizes + driver-side sampled silhouette)
 *
 * Note: Each phase runs on a newly created execution environment and re-reads the source.
 */
public final class FlinkClusteringJob {

    private static final Logger logger = LoggerFactory.getLogger(FlinkClusteringJob.class);

    public final String framework = "flink";

    /** Reporter flush period in milliseconds. */
    private static final int REPORTER_INTERVAL_MS = 200;
    private static final String REPORTER_INTERVAL = REPORTER_INTERVAL_MS + " MILLISECONDS";

    /**
     * Settle wait time before reading metrics to allow metric reporters to flush data.
     */
    private static final long METRICS_SETTLE_MS = Math.max(600L, 3L * REPORTER_INTERVAL_MS);

    public RunResult run(RunConfig config, ClusterProfile profile) {
        Instant startedAt = Instant.now();
        long t0Total = System.nanoTime();
        final int parallelism = Params.intParam(config.dataset.params, "numPartitions");

        logger.info("starting runId=" + config.runId + " algorithm=" + config.algorithm.name
            + " dataset=" + config.dataset.type + " profile=" + profile.name()
            + " parallelism=" + parallelism);

        String materializedPath = null;
        BenchmarkListener listener = new BenchmarkListener();
        clustering.benchmark.metrics.MetricsFile.reset();
        listener.startDriverSampling();

        try {
            final boolean local = profile.isLocal();
            if (!local) {
                waitForClusterReady(parallelism);
                settleForMetrics();
                listener.captureBaseline();
                t0Total = System.nanoTime();
                startedAt = Instant.now();
            }

            final boolean webUi = local && "1".equals(System.getenv("FLINK_WEB_UI"));

            if (!local && !clustering.benchmark.metrics.MetricsFile.isExplicitlyConfigured()) {
                throw new IllegalStateException(
                    "Non-local profile requires a shared metrics path visible to JobManager and "
                    + "all TaskManagers. Set CLUSTERING_METRICS_FILE (or -Dclustering.metrics.file=) "
                    + "to a shared-filesystem location. Refusing the node-local java.io.tmpdir fallback."
                );
            }

            final String metricsPath = clustering.benchmark.metrics.MetricsFile.path();
            final Map<String, String> engineConf = config.effectiveEngineConf();

            if (!engineConf.isEmpty()) {
                logger.info("applying flink_config: " + engineConf);
            }

            EnvFactory envs = () -> {
                StreamExecutionEnvironment e;
                if (local) {
                    Configuration conf = new Configuration();
                    conf.setString("metrics.reporter.file.factory.class", "clustering.metrics.FileMetricReporterFactory");
                    conf.setString("metrics.reporter.file.path", metricsPath);
                    conf.setString("metrics.reporter.file.interval", REPORTER_INTERVAL);
                    engineConf.forEach(conf::setString);

                    if (webUi) {
                        conf.set(RestOptions.PORT, 8081);
                        e = StreamExecutionEnvironment.createLocalEnvironmentWithWebUI(conf);
                    } else {
                        e = StreamExecutionEnvironment.createLocalEnvironment(parallelism, conf);
                    }
                } else {
                    Configuration conf = new Configuration();
                    engineConf.forEach(conf::setString);
                    e = StreamExecutionEnvironment.getExecutionEnvironment(conf);
                }
                e.setRuntimeMode(RuntimeExecutionMode.BATCH);
                e.setParallelism(parallelism);
                return e;
            };

            final Map<String, BenchmarkListener.PhaseDelta> phaseDeltas = new LinkedHashMap<>();
            final Map<String, Long> phaseDurationsMs = new LinkedHashMap<>();
            final Map<String, BenchmarkListener.PhasePeaks> phasePeaks = new LinkedHashMap<>();

            BenchmarkListener.Snapshot openMark = listener.mark();
            long openMs = System.currentTimeMillis();

            DataSource datasource = DataSourceRegistry.create(config.dataset);
            PointSource source = datasource::load;
            AlgorithmRegistry.Built built = AlgorithmRegistry.create(config.algorithm);
            Clusterer clusterer = built.clusterer;
            DistanceMetric evalDistance = built.distance;
            Integer nFeatures = parseIntOrNull(datasource.metadata().get("nFeatures"));

            // --- load ---
            logger.info("loading dataset " + datasource.name());
            long t0Load = System.nanoTime();
            long nRows;

            if (Boolean.parseBoolean(engineConf.getOrDefault("clustering.materialize.source", "false"))) {
                materializedPath = config.resolveOutputDir() + "/materialized/" + config.runId;
                logger.info("materializing the source to " + materializedPath);
                clustering.core.MaterializedPointSource.MaterializationResult materialized =
                    clustering.core.MaterializedPointSource.materialize(source, envs, materializedPath);
                nRows = materialized.rowCount;
                source = materialized.source;
            } else {
                nRows = Datasets.count(source, envs);
            }

            long t1Load = System.nanoTime();
            logger.info("loaded nRows=" + nRows + " in " + ms(t0Load, t1Load) + "ms");

            source = PointSource.withKnownRowCount(source, nRows);

            if (!local) {
                settleForMetrics();
            }
            BenchmarkListener.Snapshot afterLoad = listener.mark();
            long afterLoadMs = System.currentTimeMillis();
            phaseDeltas.put("load", BenchmarkListener.between(openMark, afterLoad));
            phaseDurationsMs.put("load", ms(t0Load, t1Load));
            phasePeaks.put("load", listener.windowPeaks(openMs, afterLoadMs));

            // --- fit ---
            logger.info("fitting " + config.algorithm.name);
            long t0Fit = System.nanoTime();
            Model model = clusterer.fit(source, envs, parallelism);
            long t1Fit = System.nanoTime();
            logger.info("fitted in " + ms(t0Fit, t1Fit) + "ms");

            if (!local) {
                settleForMetrics();
            }
            BenchmarkListener.Snapshot afterFit = listener.mark();
            long afterFitMs = System.currentTimeMillis();
            phaseDeltas.put("fit", BenchmarkListener.between(afterLoad, afterFit));
            phaseDurationsMs.put("fit", ms(t0Fit, t1Fit));
            phasePeaks.put("fit", listener.windowPeaks(afterLoadMs, afterFitMs));

            // --- evaluate ---
            long t0Eval = System.nanoTime();
            EvaluationResult eval = new EvaluationRunner(evalDistance)
                .run(model, source, envs, config.evaluation, parallelism);
            long t1Eval = System.nanoTime();

            logger.info("evaluated nClusters=" + eval.nClusters + " silhouette=" + eval.silhouette
                + " (scored=" + eval.silhouetteScoredPoints
                + ", sampleClusters=" + eval.silhouetteSampleClusters
                + ", unscored=" + eval.silhouetteUnscoredPoints + ")"
                + " daviesBouldin=" + eval.daviesBouldin
                + " calinskiHarabasz=" + eval.calinskiHarabasz
                + " noiseFraction=" + eval.noiseFraction + " in " + ms(t0Eval, t1Eval) + "ms");

            final long tEndWork = System.nanoTime();

            if (!local) {
                settleForMetrics();
            }

            phaseDeltas.put("eval", BenchmarkListener.between(afterFit, listener.mark()));
            phaseDurationsMs.put("eval", ms(t0Eval, t1Eval));
            phasePeaks.put("eval", listener.windowPeaks(afterFitMs, System.currentTimeMillis()));

            listener.stopDriverSampling();

            if (materializedPath != null) {
                clustering.core.MaterializedPointSource.delete(materializedPath);
            }

            RunResult r = baseResult(config, profile, startedAt, Instant.now(), "ok", null, listener.snapshot());
            applyPhases(r, phaseDeltas, phaseDurationsMs, phasePeaks);

            r.dataset = datasource.name();
            r.datasetMetadata = datasource.metadata();
            r.nRows = nRows;
            r.nPartitions = parallelism;
            r.nFeatures = nFeatures;
            r.loadDurationMs = ms(t0Load, t1Load);
            r.fitDurationMs = ms(t0Fit, t1Fit);
            r.evalDurationMs = ms(t0Eval, t1Eval);
            r.totalDurationMs = ms(t0Total, tEndWork);
            r.avgCpuCoresBusy = RunResult.avgCpuCoresBusy(r.executorCpuTimeNs, r.totalDurationMs);
            r.nClusters = eval.nClusters;
            r.noiseFraction = eval.noiseFraction;
            r.silhouette = eval.silhouette;
            r.silhouetteScoredPoints = eval.silhouetteScoredPoints;
            r.silhouetteSampleClusters = eval.silhouetteSampleClusters;
            r.silhouetteUnscoredPoints = eval.silhouetteUnscoredPoints;
            r.clusterSizes = eval.clusterSizes;
            r.daviesBouldin = eval.daviesBouldin;
            r.calinskiHarabasz = eval.calinskiHarabasz;

            return r;
        } catch (Throwable t) {
            logger.error("run failed: {}: {}", t.getClass().getSimpleName(), t.getMessage(), t);
            listener.stopDriverSampling();
            if (materializedPath != null) {
                clustering.core.MaterializedPointSource.delete(materializedPath);
            }
            RunResult r = baseResult(config, profile, startedAt, Instant.now(), "failed",
                t.getClass().getSimpleName() + ": " + t.getMessage(), listener.snapshot());
            r.dataset = config.dataset.type;
            r.datasetMetadata = new HashMap<>();
            r.nRows = -1L;
            r.nPartitions = -1;
            r.totalDurationMs = ms(t0Total, System.nanoTime());
            return r;
        }
    }

    private RunResult baseResult(
            RunConfig config,
            ClusterProfile profile,
            Instant startedAt,
            Instant finishedAt,
            String status,
            String errorMessage,
            BenchmarkListener.Snapshot snap
    ) {
        RunResult r = new RunResult();
        r.runId = config.runId;
        r.framework = framework;
        r.profile = profile.name();
        r.startedAtIso = startedAt.toString();
        r.finishedAtIso = finishedAt.toString();
        r.status = status;
        r.errorMessage = errorMessage;
        r.algorithm = config.algorithm.name;
        r.algorithmParams = flatten(config.algorithm.params);
        r.engineConf = config.effectiveEngineConf();
        r.experimentMetadata = config.experimentMetadata == null ? new HashMap<>() : config.experimentMetadata;
        r.jvmGcTimeMs = snap.jvmGcTimeMs;
        r.executorCpuTimeNs = snap.executorCpuTimeNs;
        r.cpuCoreHours = RunResult.cpuCoreHours(snap.executorCpuTimeNs);
        r.peakExecutorMemoryBytes = snap.peakExecutorMemoryBytes;
        r.totalExecutorMemoryBytes = snap.totalExecutorMemoryBytes;
        r.memoryGbHours = RunResult.memoryGbHours(snap.heapByteSeconds);
        r.peakDirectMemoryBytes = snap.peakDirectMemoryBytes;
        r.totalDirectMemoryBytes = snap.totalDirectMemoryBytes;
        r.driverCpuTimeNs = snap.driverCpuTimeNs;
        r.driverCpuCoreHours = RunResult.cpuCoreHours(snap.driverCpuTimeNs);
        r.driverPeakHeapBytes = snap.driverPeakHeapBytes;
        r.driverGcTimeMs = snap.driverGcTimeMs;
        r.driverMemoryGbHours = RunResult.memoryGbHours(snap.driverHeapByteSeconds);
        r.driverPeakDirectMemoryBytes = snap.driverPeakDirectMemoryBytes;
        return r;
    }

    private static void applyPhases(
            RunResult r,
            Map<String, BenchmarkListener.PhaseDelta> deltas,
            Map<String, Long> durationsMs,
            Map<String, BenchmarkListener.PhasePeaks> peaks
    ) {
        r.phaseDurationsMs = durationsMs;
        r.phaseWindowPeakExecutorMemoryBytes = new LinkedHashMap<>();
        r.phaseWindowTotalExecutorMemoryBytes = new LinkedHashMap<>();
        r.phaseWindowPeakDirectMemoryBytes = new LinkedHashMap<>();
        r.phaseWindowTotalDirectMemoryBytes = new LinkedHashMap<>();
        r.phaseWindowDriverPeakHeapBytes = new LinkedHashMap<>();
        r.phaseWindowDriverPeakDirectMemoryBytes = new LinkedHashMap<>();

        for (Map.Entry<String, BenchmarkListener.PhasePeaks> e : peaks.entrySet()) {
            BenchmarkListener.PhasePeaks p = e.getValue();
            r.phaseWindowPeakExecutorMemoryBytes.put(e.getKey(), p.maxWorkerHeapBytes);
            r.phaseWindowTotalExecutorMemoryBytes.put(e.getKey(), p.totalWorkerHeapBytes);
            r.phaseWindowPeakDirectMemoryBytes.put(e.getKey(), p.maxWorkerDirectBytes);
            r.phaseWindowTotalDirectMemoryBytes.put(e.getKey(), p.totalWorkerDirectBytes);
            r.phaseWindowDriverPeakHeapBytes.put(e.getKey(), p.driverHeapBytes);
            r.phaseWindowDriverPeakDirectMemoryBytes.put(e.getKey(), p.driverDirectBytes);
        }

        r.phaseCpuTimeNs = new LinkedHashMap<>();
        r.phaseGcTimeMs = new LinkedHashMap<>();
        r.phaseMemoryGbHours = new LinkedHashMap<>();
        r.phaseDriverCpuTimeNs = new LinkedHashMap<>();
        r.phaseDriverGcTimeMs = new LinkedHashMap<>();
        r.phaseDriverMemoryGbHours = new LinkedHashMap<>();

        for (Map.Entry<String, BenchmarkListener.PhaseDelta> e : deltas.entrySet()) {
            BenchmarkListener.PhaseDelta d = e.getValue();
            r.phaseCpuTimeNs.put(e.getKey(), d.cpuNanos);
            r.phaseGcTimeMs.put(e.getKey(), d.gcTimeMs);
            r.phaseMemoryGbHours.put(e.getKey(), RunResult.memoryGbHours(d.heapByteSeconds));
            r.phaseDriverCpuTimeNs.put(e.getKey(), d.driverCpuNanos);
            r.phaseDriverGcTimeMs.put(e.getKey(), d.driverGcTimeMs);
            r.phaseDriverMemoryGbHours.put(e.getKey(), RunResult.memoryGbHours(d.driverHeapByteSeconds));
        }
    }

    private static Map<String, String> flatten(Map<String, Object> params) {
        Map<String, String> out = new HashMap<>();
        if (params != null) {
            for (Map.Entry<String, Object> e : params.entrySet()) {
                out.put(e.getKey(), e.getValue() == null ? "null" : e.getValue().toString());
            }
        }
        return out;
    }

    private static Integer parseIntOrNull(String s) {
        try {
            return s == null ? null : Integer.valueOf(s);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static long ms(long startNanos, long endNanos) {
        return (endNanos - startNanos) / 1_000_000L;
    }

    /**
     * Polls the JobManager REST API to verify that enough TaskManager slots
     * have been registered for the execution environment.
     */
    private static void waitForClusterReady(int expectedSlots) throws Exception {
        HttpClient client = HttpClient.newHttpClient();
        HttpRequest req = HttpRequest.newBuilder(URI.create("http://localhost:8081/overview")).GET().build();
        ObjectMapper mapper = new ObjectMapper();

        for (int attempt = 1; attempt <= 60; attempt++) {
            try {
                HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
                JsonNode node = mapper.readTree(resp.body());
                int slotsTotal = node.path("slots-total").asInt(0);
                if (slotsTotal >= expectedSlots) {
                    logger.info("cluster ready: " + slotsTotal + "/" + expectedSlots
                        + " slots after " + attempt + " attempt(s)");
                    return;
                }
            } catch (Exception e) {
                // Keep polling
            }
            Thread.sleep(2000L);
        }
        throw new IllegalStateException(
            "TaskManagers did not register " + expectedSlots + " slots within 120s of JobManager startup"
        );
    }

    /**
     * Sleeps for the predefined metric settling duration to allow reporters to flush.
     */
    private static void settleForMetrics() {
        logger.info("settling " + METRICS_SETTLE_MS + "ms for metric reporters to flush");
        try {
            Thread.sleep(METRICS_SETTLE_MS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
