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

/** Runs one clustering benchmark on Flink. Java mirror of the Spark repo's
 *  {@code clustering.benchmark.SparkClusteringJob} (no shared interface — each engine
 *  repo owns its single job class).
 *
 *  Pipeline: load (distributed count) -> fit (distributed, see the chosen algorithm) ->
 *  evaluate (distributed cluster sizes + driver-side sampled silhouette). Every Flink action
 *  runs on a fresh env from the factory, and each one re-reads the source — see the load
 *  phase for what that costs against Spark's cached DataFrame, and why the materialize
 *  variant that removed the cost was reverted.
 *
 *  CLUSTER CAVEAT: this program issues several Flink jobs per run (count, fit, the
 *  sampling jobs some algorithms need, sizes). On the local MiniCluster each is a fresh
 *  independent env. On a real cluster, multi-job-per-submission via getExecutionEnvironment()
 *  must be validated; every iterative algorithm here runs its fit as a SINGLE job (FLIP-176),
 *  which is the cluster-friendly path. */
public final class FlinkClusteringJob {

    private static final Logger logger = LoggerFactory.getLogger(FlinkClusteringJob.class);

    /** Engine tag written into every RunResult (mirrors Spark's {@code val framework = "spark"}). */
    public final String framework = "flink";

    /** Reporter flush period (also the {@code metrics.reporter.file.interval} value, which on a
     *  real cluster comes from the sbatch template's flink-conf instead of from here).
     *
     *  <p>200 ms rather than 1 s, matching the Spark plugin's
     *  {@code spark.clustering.metrics.samplingIntervalMs} so both engines' counters are equally
     *  fresh. Two things get better: a phase boundary is sharp to ~200 ms instead of ~1 s, and
     *  the settle derived from it (see {@link #METRICS_SETTLE_MS}) shrinks with it — wall clock
     *  the run does not have to spend. The cost is 5 flushes/s per process instead of 1: the
     *  file holds ~10 lines now that the unread {@code numBytes*} counters are gone, so the
     *  bandwidth is nothing, but each flush is a write + rename on the SHARED filesystem, i.e.
     *  two metadata ops. At ~100 TaskManagers that is ~1000 Lustre metadata ops/s for the run's
     *  duration. Raise the interval in flink-conf if that ever shows up next to the job's own
     *  I/O — the Spark side pays an RPC instead of a file write, so this is the one place where
     *  the two engines' instrumentation cost is NOT symmetric. */
    private static final int REPORTER_INTERVAL_MS = 200;
    private static final String REPORTER_INTERVAL = REPORTER_INTERVAL_MS + " MILLISECONDS";
    /** Settle wait before reading metrics on a real cluster: must exceed the reporter
     *  interval so the post-teardown {@link clustering.metrics.FileMetricReporter#report()}
     *  tick flushes every TaskManager's final values to the shared file before the driver
     *  aggregates. There is no cross-process flush barrier, so this is the safety margin.
     *  Three reporter periods, floored at 600 ms — same derivation as the Spark side's
     *  {@code ProcessCpuPlugin.settleMs}. */
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
        clustering.benchmark.metrics.MetricsFile.reset();  // clear last run's metrics file
        // The driver is THIS process in both profiles: locally it is the only JVM, and on the
        // cluster it is the `flink run` client that session mode puts main() in (see
        // DriverProcess for why the bootstrap moved off Application Mode). Started before the
        // baseline so the first phase already has samples behind it.
        listener.startDriverSampling();
        try {
            final boolean local = profile.isLocal();
            if (!local) {
                // Under Application Mode, main() (this method) runs INSIDE the JobManager
                // process, and starts as soon as the JM is up — NOT once TaskManagers have
                // registered (the sbatch template's wait_for_taskmanagers is a slower,
                // cluster-wide backstop that races this same registration concurrently, it
                // cannot block anything already running inside this JVM). Without this gate the
                // first job below would silently block in Flink's own scheduler waiting for
                // slots, and that wait would leak into loadDurationMs/totalDurationMs — unlike
                // Spark, whose driver JVM only starts after its SLURM-side worker gate already
                // passed. Reset t0Total after the gate so the measured window matches Spark's.
                waitForClusterReady(parallelism);
                // Rebase Flink's cumulative-since-JVM-start CPU/GC counters onto the run
                // window, the way Spark's ProcessCpuPlugin already reports a delta. Settle
                // first so every TaskManager has flushed at least one tick; both the settle
                // and the read sit BEFORE the clock starts, so they cost the measurement
                // nothing. See BenchmarkListener#captureBaseline.
                settleForMetrics();
                listener.captureBaseline();
                t0Total = System.nanoTime();
                // Keep the ISO stamps on the same window as totalDurationMs; left at the
                // process's own start they disagreed by the whole registration wait, so
                // finishedAtIso - startedAtIso and totalDurationMs answered different
                // questions depending on which engine wrote the result.
                startedAt = Instant.now();
            }
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
            // Per-run engine configuration from the YAML's `flink_config`. It used to be
            // dropped on the floor by RunConfig's Jackson binding while Spark's `spark_config`
            // was applied — one engine had a working tuning escape hatch and the other only
            // appeared to. Job-scoped keys (execution.*, pipeline.*, taskmanager.memory.managed
            // .consumer-weights, ...) take effect here; TaskManager PROCESS-level keys cannot be
            // set per job and belong in the sbatch template's flink-conf.yaml instead.
            final Map<String, String> engineConf = config.effectiveEngineConf();
            if (!engineConf.isEmpty()) {
                logger.info("applying flink_config: " + engineConf);
            }
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

            // Phase boundaries start here, at the same instant the run-level baseline was taken
            // (or at zero locally, where MetricsFile.reset() has just emptied the files), so the
            // first phase is measured from the same point the run-level totals are.
            final Map<String, BenchmarkListener.PhaseDelta> phaseDeltas = new LinkedHashMap<>();
            final Map<String, Long> phaseDurationsMs = new LinkedHashMap<>();
            final Map<String, BenchmarkListener.PhasePeaks> phasePeaks = new LinkedHashMap<>();
            BenchmarkListener.Snapshot openMark = listener.mark();
            // Wall clock, not nanoTime: the sampled peaks are bucketed against the reporters'
            // own System.currentTimeMillis stamps (see MetricsFile.windowPeaks).
            long openMs = System.currentTimeMillis();

            DataSource datasource = DataSourceRegistry.create(config.dataset);
            PointSource source = datasource::load;
            AlgorithmRegistry.Built built = AlgorithmRegistry.create(config.algorithm);
            Clusterer clusterer = built.clusterer;
            // The metric the clusterer actually ran with — for the k-means family that is the
            // one implied by `geometry`, not a config `distance` field (mirrors Spark's Built).
            DistanceMetric evalDistance = built.distance;
            Integer nFeatures = parseIntOrNull(datasource.metadata().get("nFeatures"));

            // --- load (distributed count) -------------------------------------
            // Reverted from the materialize-to-shared-storage variant (29.08.2026): its
            // FileSink job (`materialize-source`) dies on any multi-node allocation with a
            // SerializedLambda -> CommittableMessageTypeInfo ClassCastException — 1 node OK,
            // 3 and 4 nodes FAILED — so it blocked every distributed Flink run. See
            // clustering/core/MaterializedPointSource.java, which is kept but no longer wired
            // in, and the pom note on flink-connector-files for what was ruled out.
            //
            // What the revert gives up, and it is a real measurement asymmetry to state rather
            // than forget: every algorithm mints a fresh env per internal job (EnvFactory's
            // doc), and several are BY DESIGN multiple sequential jobs (CLARA/PAMAE/dbscanpp's
            // P1/P2 phases), so each of them re-reads the ORIGINAL source. Against Spark's
            // persist(MEMORY_AND_DISK)+count() — one read, then cache — that means Flink's
            // fitDurationMs/evalDurationMs silently re-pay connector I/O that Spark does not.
            // Invisible on `synthetic` (generated per read, no I/O to re-pay) and on anything
            // small enough to sit in page cache; it bites on the parquet datasets, hardest on
            // Cohere. Do NOT compare Flink and Spark fit times on parquet input until this is
            // solved — by fixing the materialize path or by another route to one read per run.
            logger.info("loading dataset " + datasource.name());
            long t0Load = System.nanoTime();
            long nRows;
            // `flink_config: {clustering.materialize.source: "true"}` in the experiment YAML turns
            // the load phase into Spark's `persist(...)+count()`: read
            // the ORIGINAL source once, write it to shared storage as parquet, and hand every
            // later phase a source that reads THAT copy. Off by default while it is being
            // measured; the switch is an env var rather than a config field so one jar can run
            // both arms of the comparison.
            //
            // Why it is worth re-testing now (5.09.2026): this path was written in August and
            // reverted because its FileSink job died on every multi-node allocation with
            // `SerializedLambda -> ClassCastException`. That failure has since been diagnosed —
            // it was Application Mode shipping the jar to TaskManagers as a `usrlib` PATH, and the
            // bootstrap moved to session mode (jar as a BlobServer blob), which fixed the same
            // failure everywhere else it occurred. So the reason for the revert may simply be gone.
            //
            // The prize is larger than "one read instead of three": with `sampleFraction`, re-
            // reading the source rescans every file of the dataset to discard most rows, while the
            // materialized copy holds only what survived — on Gaia at 1% that is ~280 MB against a
            // full scan of 26 GB across 3118 files, twice per run.
            // Read from the run's engineConf rather than an env var, so the switch travels the
            // same path as every other per-run Flink setting and the harness needs no special case.
            if (Boolean.parseBoolean(engineConf.getOrDefault("clustering.materialize.source", "false"))) {
                materializedPath = config.resolveOutputDir() + "/materialized/" + config.runId;
                logger.info("materializing the source to " + materializedPath);
                clustering.core.MaterializedPointSource.Result materialized =
                    clustering.core.MaterializedPointSource.materialize(source, envs, materializedPath);
                // The row count comes from the write's own accumulator, so this replaces the count
                // job outright rather than adding to it.
                nRows = materialized.nRows;
                source = materialized.source;
            } else {
                nRows = Datasets.count(source, envs);
            }
            long t1Load = System.nanoTime();
            logger.info("loaded nRows=" + nRows + " in " + ms(t0Load, t1Load) + "ms");

            // Hand the counted size DOWN with the source: `clara` and `dbscanpp` need n, and
            // without this they each submit a second count job that re-reads the whole dataset to
            // learn the number just logged above. Everything below uses the counting source; a
            // derived source (a filter, a sample) is a different instance and reports no count, so
            // the number can never leak onto a stream that does not have those rows.
            source = PointSource.withKnownRowCount(source, nRows);

            // Boundary AFTER the phase's duration was taken, so the settle it waits out cannot
            // inflate the phase's wall clock — only its (idle, so ~zero) CPU. Skipped locally:
            // each phase's MiniCluster flushes its reporter on close(), so the mark is already
            // past the last write.
            if (!local) {
                settleForMetrics();
            }
            BenchmarkListener.Snapshot afterLoad = listener.mark();
            long afterLoadMs = System.currentTimeMillis();
            phaseDeltas.put("load", BenchmarkListener.between(openMark, afterLoad));
            phaseDurationsMs.put("load", ms(t0Load, t1Load));
            phasePeaks.put("load", listener.windowPeaks(openMs, afterLoadMs));

            // --- fit ----------------------------------------------------------
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

            // --- evaluate (see EvaluationRunner; only the requested metrics are computed) ---
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

            // Stop the clock BEFORE the metrics settle below. The settle is instrumentation, not
            // work: leaving it inside the window charged every cluster run a flat +2s that Spark
            // never pays, which on smoke-scale runs is most of the measurement.
            final long tEndWork = System.nanoTime();

            // No cross-process flush barrier: on a real cluster wait one reporter tick (+margin)
            // after the last job so every TaskManager flushes its final values to the shared file
            // before we aggregate. Skipped locally (same JVM, captured by the periodic ticks).
            if (!local) {
                settleForMetrics();
            }
            // Closed after the post-work settle, so the phases add up to the run-level totals.
            phaseDeltas.put("eval", BenchmarkListener.between(afterFit, listener.mark()));
            phaseDurationsMs.put("eval", ms(t0Eval, t1Eval));
            phasePeaks.put("eval", listener.windowPeaks(afterFitMs, System.currentTimeMillis()));

            // Final driver reading before the snapshot, so the tail of `eval` is not lost.
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
            // eval fields stay null on a failed run -> omitted (matches Spark's empty eval).
            return r;
        }
    }

    private RunResult baseResult(RunConfig config, ClusterProfile profile, Instant startedAt,
                                 Instant finishedAt, String status, String errorMessage,
                                 BenchmarkListener.Snapshot snap) {
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

    /** Spread the per-phase deltas across the RunResult's parallel maps, converting heap
     *  byte-seconds to GB-hours so the per-phase unit matches {@code memoryGbHours}. */
    private static void applyPhases(RunResult r,
                                    Map<String, BenchmarkListener.PhaseDelta> deltas,
                                    Map<String, Long> durationsMs,
                                    Map<String, BenchmarkListener.PhasePeaks> peaks) {
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

    /** Poll this process's own REST server — main() runs inside the JobManager under Application
     *  Mode, so it always has one at localhost:8081 — until enough TaskManager slots have
     *  registered to run the job at {@code expectedSlots} parallelism. 60 attempts x 2s mirrors
     *  the timeout budget of the sbatch template's own wait_for_taskmanagers. */
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
                // REST server may not be bound yet on the first few attempts - keep polling.
            }
            Thread.sleep(2000L);
        }
        throw new IllegalStateException(
            "TaskManagers did not register " + expectedSlots + " slots within 120s of JobManager startup");
    }

    /** Wait for reporters to flush final metrics to the shared file (see {@link #METRICS_SETTLE_MS}). */
    private static void settleForMetrics() {
        logger.info("settling " + METRICS_SETTLE_MS + "ms for metric reporters to flush");
        try {
            Thread.sleep(METRICS_SETTLE_MS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
