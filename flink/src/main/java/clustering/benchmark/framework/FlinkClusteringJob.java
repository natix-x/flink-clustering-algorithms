package clustering.benchmark.framework;

import clustering.benchmark.config.ClusterProfile;
import clustering.benchmark.config.Params;
import clustering.benchmark.config.RunConfig;
import clustering.benchmark.datasource.DataSource;
import clustering.benchmark.evaluation.LabelCount;
import clustering.benchmark.metrics.BenchmarkListener;
import clustering.benchmark.metrics.RunResult;
import clustering.benchmark.registry.AlgorithmRegistry;
import clustering.benchmark.registry.DistanceRegistry;
import clustering.core.Clusterer;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import clustering.distance.EuclideanDistance;
import clustering.evaluation.SilhouetteEvaluator;
import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.time.Instant;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Flink-backed {@link ClusteringJob}.
 *
 *  Pipeline: count (load) -> fit (distributed, see the chosen KMeans impl) ->
 *  evaluate (distributed cluster sizes + driver-side sampled silhouette). Each
 *  Flink action runs on a fresh env from the factory.
 *
 *  CLUSTER CAVEAT: this program issues several Flink jobs per run (count, the
 *  per-iteration jobs of impl=jobiter, sizes). On the local MiniCluster each is a
 *  fresh independent env. On a real cluster, multi-job-per-submission via
 *  getExecutionEnvironment() must be validated; impl=mliter runs fit as a single
 *  job and is the cluster-friendly path. */
public final class FlinkClusteringJob implements ClusteringJob {

    /** Reporter flush period (also the {@code metrics.reporter.file.interval} value). */
    private static final String REPORTER_INTERVAL = "1 SECONDS";
    /** Settle wait before reading metrics on a real cluster: must exceed the reporter
     *  interval so the post-teardown {@link clustering.metrics.FileMetricReporter#report()}
     *  tick flushes every TaskManager's final values to the shared file before the driver
     *  aggregates. There is no cross-process flush barrier, so this is the safety margin. */
    private static final long METRICS_SETTLE_MS = 2_000L;

    @Override
    public String framework() {
        return "flink";
    }

    @Override
    public RunResult run(RunConfig config, ClusterProfile profile) {
        Instant startedAt = Instant.now();
        long t0Total = System.nanoTime();
        final int parallelism = Params.intParam(config.dataset.params, "numPartitions");
        log("starting runId=" + config.runId + " algorithm=" + config.algorithm.name
            + " dataset=" + config.dataset.type + " profile=" + profile.name()
            + " parallelism=" + parallelism);

        BenchmarkListener listener = new BenchmarkListener();
        clustering.benchmark.metrics.MetricsFile.reset();  // clear last run's metrics file
        try {
            final boolean local = profile.isLocal();
            final boolean webUi = local && "1".equals(System.getenv("FLINK_WEB_UI"));
            // On a real cluster JM + TMs are separate processes/nodes; each reporter writes to
            // its own local disk unless the path is a SHARED filesystem. Refuse the node-local
            // tmpdir fallback there, or the driver would silently aggregate only its own node.
            if (!local && !clustering.benchmark.metrics.MetricsFile.isExplicitlyConfigured()) {
                throw new IllegalStateException(
                    "Non-local profile requires a shared metrics path visible to JobManager and "
                    + "all TaskManagers. Set CLUSTERING_METRICS_FILE (or -Dclustering.metrics.file=) "
                    + "to a shared-filesystem location (e.g. $SCRATCH/flink-metrics.txt). "
                    + "Refusing the node-local java.io.tmpdir fallback.");
            }
            final String metricsPath = clustering.benchmark.metrics.MetricsFile.path();
            EnvFactory envs = () -> {
                StreamExecutionEnvironment e;
                if (local) {
                    // Load the file metric reporter into the local MiniCluster so engine
                    // metrics land in the same file the driver reads (on a real cluster the
                    // reporter is configured in flink-conf instead).
                    Configuration conf = new Configuration();
                    conf.setString("metrics.reporter.file.factory.class",
                        "clustering.metrics.FileMetricReporterFactory");
                    conf.setString("metrics.reporter.file.path", metricsPath);
                    conf.setString("metrics.reporter.file.interval", REPORTER_INTERVAL);
                    if (webUi) {
                        conf.set(RestOptions.PORT, 8081);
                        e = StreamExecutionEnvironment.createLocalEnvironmentWithWebUI(conf);
                    } else {
                        e = StreamExecutionEnvironment.createLocalEnvironment(parallelism, conf);
                    }
                } else {
                    e = StreamExecutionEnvironment.getExecutionEnvironment();
                }
                e.setRuntimeMode(RuntimeExecutionMode.BATCH);
                e.setParallelism(parallelism);
                return e;
            };

            DataSource datasource = DataSource.create(config.dataset);
            PointSource source = datasource::load;
            Clusterer clusterer = AlgorithmRegistry.create(config.algorithm);
            DistanceMetric evalDistance = evaluationDistance(config);
            Integer nFeatures = parseIntOrNull(datasource.metadata().get("nFeatures"));

            // --- load (distributed count) -------------------------------------
            log("loading dataset " + datasource.name());
            long t0Load = System.nanoTime();
            long nRows = Datasets.count(source, envs);
            long t1Load = System.nanoTime();
            log("loaded nRows=" + nRows + " in " + ms(t0Load, t1Load) + "ms");

            // --- fit ----------------------------------------------------------
            log("fitting " + config.algorithm.name);
            long t0Fit = System.nanoTime();
            Model model = clusterer.fit(source, envs, parallelism);
            long t1Fit = System.nanoTime();
            log("fitted in " + ms(t0Fit, t1Fit) + "ms");

            // --- evaluate (distributed sizes + sampled silhouette) ------------
            // Only the metrics named in EvaluationSpec.metrics are computed; the rest stay
            // null and are omitted from the result (mirrors Spark's EvaluationRunner).
            // clusterSizes/noiseFraction/nClusters all derive from one assignment scan, so
            // run it once iff at least one of them is requested.
            long t0Eval = System.nanoTime();
            boolean wantSizes = config.evaluation.wants("clusterSizes");
            boolean wantNoise = config.evaluation.wants("noiseFraction");
            boolean wantNClusters = config.evaluation.wants("nClusters");
            Map<String, Long> sizes = (wantSizes || wantNoise || wantNClusters)
                ? computeSizes(model, source, envs) : null;
            Double noiseFraction = null;
            if (wantNoise) {
                long noise = sizes.getOrDefault("-1", 0L);
                noiseFraction = nRows == 0 ? 0.0 : (double) noise / nRows;
            }
            Integer nClusters = null;
            if (wantNClusters) {
                int n = 0;
                for (String key : sizes.keySet()) {
                    if (Integer.parseInt(key) >= 0) n++;
                }
                nClusters = n;
            }
            Double silhouette = null;
            if (config.evaluation.wants("silhouette")) {
                long cap = config.evaluation.sampleSize != null ? config.evaluation.sampleSize : 10_000L;
                List<double[]> sample = Datasets.collectHead(source, envs, cap);
                silhouette = new SilhouetteEvaluator(evalDistance).evaluate(sample, model.labels(sample));
            }
            long t1Eval = System.nanoTime();
            log("evaluated nClusters=" + nClusters + " silhouette=" + silhouette
                + " noiseFraction=" + noiseFraction + " in " + ms(t0Eval, t1Eval) + "ms");

            // No cross-process flush barrier: on a real cluster wait one reporter tick (+margin)
            // after the last job so every TaskManager flushes its final values to the shared file
            // before we aggregate. Skipped locally (same JVM, captured by the periodic ticks).
            if (!local) {
                settleForMetrics();
            }
            RunResult r = baseResult(config, profile, startedAt, Instant.now(), "ok", null, listener.snapshot());
            r.dataset = datasource.name();
            r.datasetMetadata = datasource.metadata();
            r.nRows = nRows;
            r.nPartitions = parallelism;
            r.nFeatures = nFeatures;
            r.loadDurationMs = ms(t0Load, t1Load);
            r.fitDurationMs = ms(t0Fit, t1Fit);
            r.evalDurationMs = ms(t0Eval, t1Eval);
            r.totalDurationMs = ms(t0Total, System.nanoTime());
            r.avgCpuCoresBusy = RunResult.avgCpuCoresBusy(r.executorCpuTimeNs, r.totalDurationMs);
            r.nClusters = nClusters;
            r.noiseFraction = noiseFraction;
            r.silhouette = silhouette;
            r.clusterSizes = wantSizes ? sizes : null;
            return r;
        } catch (Throwable t) {
            log("run failed: " + t.getClass().getSimpleName() + ": " + t.getMessage());
            RunResult r = baseResult(config, profile, startedAt, Instant.now(), "failed",
                t.getClass().getSimpleName() + ": " + t.getMessage(), listener.snapshot());
            r.dataset = config.dataset.type;
            r.datasetMetadata = new HashMap<>();
            r.nRows = -1L;
            r.nPartitions = -1;
            r.totalDurationMs = ms(t0Total, System.nanoTime());
            // eval fields stay null on a failed run -> omitted (matches Spark's empty eval).
            return r;
        }
    }

    /** Distributed cluster sizes: label every point, count per label. */
    private Map<String, Long> computeSizes(Model model, PointSource source, EnvFactory envs) {
        StreamExecutionEnvironment env = envs.newEnv();
        DataStream<LabelCount> counts = source.create(env)
            .map(new LabelMap(model)).returns(LabelCount.class)
            .keyBy(c -> c.label)
            .reduce((a, b) -> {
                LabelCount r = new LabelCount();
                r.label = a.label;
                r.count = a.count + b.count;
                return r;
            });
        Map<String, Long> sizes = new LinkedHashMap<>();
        for (LabelCount c : clustering.core.FlinkJobs.collectAll(counts, "cluster-sizes")) {
            sizes.put(Integer.toString(c.label), c.count);
        }
        return sizes;
    }

    static final class LabelMap implements MapFunction<double[], LabelCount> {
        private final Model model;

        LabelMap(Model model) {
            this.model = model;
        }

        @Override
        public LabelCount map(double[] p) {
            LabelCount c = new LabelCount();
            c.label = model.predict(p);
            c.count = 1L;
            return c;
        }
    }

    private DistanceMetric evaluationDistance(RunConfig config) {
        Object d = config.algorithm.params.get("distance");
        return d == null ? EuclideanDistance.INSTANCE : DistanceRegistry.get(d.toString());
    }

    private RunResult baseResult(RunConfig config, ClusterProfile profile, Instant startedAt,
                                 Instant finishedAt, String status, String errorMessage,
                                 BenchmarkListener.Snapshot snap) {
        RunResult r = new RunResult();
        r.runId = config.runId;
        r.framework = framework();
        r.profile = profile.name();
        r.startedAtIso = startedAt.toString();
        r.finishedAtIso = finishedAt.toString();
        r.status = status;
        r.errorMessage = errorMessage;
        r.algorithm = config.algorithm.name;
        r.algorithmParams = flatten(config.algorithm.params);
        r.engineConf = config.effectiveEngineConf();
        r.experimentMetadata = config.experimentMetadata == null ? new HashMap<>() : config.experimentMetadata;
        r.shuffleReadBytes = snap.shuffleReadBytes;
        r.shuffleWriteBytes = snap.shuffleWriteBytes;
        r.inputBytes = snap.inputBytes;
        r.outputBytes = snap.outputBytes;
        r.jvmGcTimeMs = snap.jvmGcTimeMs;
        r.executorCpuTimeNs = snap.executorCpuTimeNs;
        r.taskCount = snap.taskCount;
        r.peakExecutorMemoryBytes = snap.peakExecutorMemoryBytes;
        r.totalExecutorMemoryBytes = snap.totalExecutorMemoryBytes;
        // diskBytesSpilled, memoryBytesSpilled, shuffleFetchWaitTimeMs, shuffleWriteTimeNs,
        // failedTaskCount, stageCount, totalStageMs: no Flink analogue -> left null = N/A.
        return r;
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

    /** Wait for reporters to flush final metrics to the shared file (see {@link #METRICS_SETTLE_MS}). */
    private static void settleForMetrics() {
        log("settling " + METRICS_SETTLE_MS + "ms for metric reporters to flush");
        try {
            Thread.sleep(METRICS_SETTLE_MS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static void log(String msg) {
        System.out.println("[FlinkClusteringJob] " + msg);
    }
}