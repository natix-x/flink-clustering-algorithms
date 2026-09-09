package clustering.benchmark.evaluation;

import clustering.benchmark.config.EvaluationSpec;
import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.core.WeightedPoint;
import clustering.core.Weights;
import clustering.distance.DistanceMetric;
import clustering.evaluation.CalinskiHarabaszIndex;
import clustering.evaluation.ClusterMoments;
import clustering.evaluation.DaviesBouldinIndex;
import clustering.evaluation.SilhouetteEvaluator;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;


/** Post-fit metrics. Java mirror of the Spark {@code clustering.benchmark.evaluation.
 *  EvaluationRunner} — same metric names, same sharing, same degenerate cases, so the two engines'
 *  quality columns are comparable field by field.
 *
 *  <p>Known limitation, to be revisited when the {@code coreset} entry lands: {@code clusterSizes}
 *  and {@code noiseFraction} count ROWS, not weight mass ({@link Weights}). On unweighted input —
 *  every dataset in the matrix today — the two coincide. On a weighted (already reduced) input they
 *  do not, and the numbers then describe the reduced set rather than the data it stands for. The
 *  three quality indices are weight-aware: {@code silhouette}, {@code daviesBouldin} and
 *  {@code calinskiHarabasz} all aggregate MASS. Shared with Spark on purpose — diverging here would
 *  make the two engines' quality numbers incomparable, which is worse than a shared gap.
 *
 *  <p>The silhouette's SUBSAMPLE is mass-proportional too, so on weighted input it samples the
 *  population the stream stands for, not the reduced set. It is drawn inside the evaluator, after
 *  the noise filter, so {@code sampleSize} means the same number of SCORED points on every
 *  algorithm.
 *
 *  <h3>Passes over the data</h3>
 *  Each block below labels the full dataset independently, because a Flink job cannot see another
 *  job's results and there is no cross-job cache: label stats = 1 pass, centroid moments = 2,
 *  silhouette = 1. Spark pays the same count of LABELLINGS but reads its cached input rather than
 *  the source, so on the parquet datasets Flink re-pays connector I/O here that Spark does not —
 *  the same asymmetry {@code FlinkClusteringJob} documents for the fit phase.
 *
 *  <p>What IS avoided is a pass that buys nothing: the label-stats scan runs whenever the
 *  silhouette is requested, even if no label metric was, because it yields BOTH the draw's
 *  denominator and the per-cluster masses the evaluator needs to tell a true singleton from an
 *  under-drawn cluster. Deriving only the total from a separate mass job costs the same scan and
 *  returns strictly less. */
public final class EvaluationRunner {

    /** Every metric name the config may name. A typo used to yield a result with the field quietly
     *  absent — a run that silently measured less than the config asked for is worse than one that
     *  refuses to start. Same set, same rule, as Spark. */
    public static final Set<String> KNOWN_METRICS = new HashSet<>(Arrays.asList(
        "silhouette", "nClusters", "clusterSizes", "noiseFraction", "daviesBouldin", "calinskiHarabasz"));

    /** Most labelled ROWS an unsampled ({@code "sampleSize": null}) silhouette may collect to the
     *  driver. Rows rather than mass because rows are what the driver holds: a 100-row coreset of
     *  mass 10^9 collects 100 rows and is perfectly safe, so guarding on mass would refuse exactly
     *  the input the full silhouette is cheapest on.
     *
     *  <p>Deliberately not a knob, and the same number as Spark's: it marks where BOTH terms are
     *  still merely expensive. At 50 000 rows the driver array is 3 MB at 8-D and 400 MB at 1024-D
     *  (Cohere, the widest in the matrix), and the O(n²·d) scoring is ~10^10 flops at 8-D and
     *  ~2·10^12 at 1024-D — the cost term bites well before the memory one. */
    public static final long FULL_SILHOUETTE_MAX_ROWS = 50_000L;

    /** Label of a noise point. */
    private static final int NOISE_LABEL = -1;

    private final DistanceMetric distance;

    public EvaluationRunner(DistanceMetric distance) {
        this.distance = distance;
    }

    /** @param parallelism cluster parallelism the run was configured with; the silhouette uses it
     *                     to decide whether its m² scoring is worth its own Flink job. */
    public EvaluationResult run(Model model, PointSource source, EnvFactory envs,
                                EvaluationSpec spec, int parallelism) {
        Set<String> metrics = new HashSet<>(spec.metrics == null ? Collections.emptyList() : spec.metrics);
        if (!KNOWN_METRICS.containsAll(metrics)) {
            Set<String> unknown = new TreeSet<>(metrics);
            unknown.removeAll(KNOWN_METRICS);
            throw new IllegalArgumentException("unknown evaluation metric(s): "
                + String.join(", ", unknown) + "; known: "
                + String.join(", ", new TreeSet<>(KNOWN_METRICS)));
        }

        boolean wantSilhouette = metrics.contains("silhouette");
        boolean wantSizes = metrics.contains("clusterSizes");
        boolean wantNoise = metrics.contains("noiseFraction");
        boolean wantNClusters = metrics.contains("nClusters");
        boolean wantDaviesBouldin = metrics.contains("daviesBouldin");
        boolean wantCalinskiHarabasz = metrics.contains("calinskiHarabasz");

        // clusterSizes, noiseFraction and nClusters all derive from one labelling scan; run it once
        // iff at least one of them is requested. The silhouette joins that list even though it
        // reports none of the three: the same scan gives it the de-noised total mass (the draw's
        // denominator) AND the per-cluster masses.
        //
        // `nClusters` is the ONE exception to "only what was asked for": requesting the silhouette
        // requests it implicitly, because `silhouetteSampleClusters` is a bare number until it is
        // read against the full-data cluster count — "87 clusters in the sample" says nothing until
        // you know whether the labelling had 87 or 100, and a shortfall is exactly the case where
        // the score reads HIGH. The scan that answers it has already run by then.
        LabelStats stats = (wantSizes || wantNoise || wantNClusters || wantSilhouette)
            ? computeStats(model, source, envs) : null;

        // Both centroid indices are derived from the SAME per-cluster moments (centroids, masses,
        // distance sums), so the two passes behind them run once for the pair. Full data, never a
        // sample: unlike the silhouette they are linear in n, and sampling would only add variance
        // to a number that costs one scan to get exactly.
        ClusterMoments moments = (wantDaviesBouldin || wantCalinskiHarabasz)
            ? ClusterMoments.compute(model, source, envs, distance) : null;

        SilhouetteEvaluator.Outcome silhouette = wantSilhouette
            ? computeSilhouette(model, source, envs, spec, parallelism, stats) : null;

        return new EvaluationResult(
            // Not gated on `wantNClusters`: emitted whenever the scan ran at all, because
            // `silhouetteSampleClusters` is unreadable without it, and the count is already here.
            stats == null ? null : stats.clusterCount(),
            wantNoise && stats != null ? stats.noiseFraction() : null,
            silhouette == null ? null : silhouette.score,
            silhouette == null ? null : silhouette.scoredPoints,
            silhouette == null ? null : silhouette.sampleClusters,
            silhouette == null ? null : silhouette.unscoredPoints,
            wantSizes && stats != null ? stats.sizesByLabel() : null,
            wantDaviesBouldin && moments != null ? DaviesBouldinIndex.of(moments, distance) : null,
            wantCalinskiHarabasz && moments != null ? CalinskiHarabaszIndex.of(moments, distance) : null);
    }

    /** Silhouette on the full data set or a subsample, plus the counters that say what it was
     *  actually computed on. Sampling is essential for big data — the underlying scoring is O(n²).
     *
     *  <p>The draw itself lives in the evaluator, which applies it AFTER dropping noise so that
     *  {@code sampleSize} counts scored points rather than pre-filter rows. The de-noised
     *  per-cluster masses come from the label-stats scan, which is always run when the silhouette is
     *  requested, so the evaluator needs no counting job of its own — the silhouette then costs
     *  exactly one pass — and can tell a genuine singleton cluster (score 0, Rousseeuw) from one the
     *  sample happened to under-draw (unscorable, excluded and counted). */
    private SilhouetteEvaluator.Outcome computeSilhouette(
            Model model, PointSource source, EnvFactory envs, EvaluationSpec spec,
            int parallelism, LabelStats stats) {
        SilhouetteEvaluator.Population population =
            stats == null ? null : new SilhouetteEvaluator.Population(stats.clusteredMass());

        // `"sampleSize": null` in the config means "no sampling", which collects the WHOLE
        // de-noised dataset to the driver — on Gaia an OOM that kills the run mid-matrix, after the
        // fit has already been paid for. Refuse it here, in the same spirit as the unknown-metric
        // check above: a run that cannot produce the number it was asked for should not start.
        // Both numbers come off the label-stats scan, so the check costs nothing.
        long labelledRows = stats == null ? Long.MAX_VALUE : stats.clusteredRows();
        boolean sampled = spec.sampleSize() != null && spec.sampleSize() > 0;
        if (!sampled && labelledRows > FULL_SILHOUETTE_MAX_ROWS) {
            throw new IllegalArgumentException("silhouette requested with no sampleSize on "
                + labelledRows + " labelled rows, above the " + FULL_SILHOUETTE_MAX_ROWS
                + " the driver can collect: an unsampled silhouette collects every labelled row and "
                + "is O(n^2 * d). Set evaluation.sampleSize (10000 is the default) or run this "
                + "config on a smaller dataset.");
        }

        return new SilhouetteEvaluator(distance)
            .measure(model, source, envs, parallelism, spec.sampleSize(), spec.seed, population);
    }

    /** Cluster id -> (row count, mass), from a single distributed scan.
     *
     *  The mass rides along on the same scan because the silhouette's subsample needs it and a
     *  second sum-of-weights pass would mean labelling the whole dataset twice. On unweighted input
     *  every weight is 1.0, so mass == row count.
     *
     *  <p>{@link Weights#safe} rather than the raw weight: a corrupt weight must not be able to make
     *  a cluster's mass NaN or negative here either, or the mass the silhouette divides by would
     *  disagree with the mass it sums. */
    private LabelStats computeStats(Model model, PointSource source, EnvFactory envs) {
        StreamExecutionEnvironment env = envs.newEnv();
        DataStream<LabelCount> counts = source.create(env)
            .map(new LabelMap(model)).returns(LabelCount.class)
            .keyBy(lc -> lc.label)
            .reduce((a, b) -> {
                LabelCount r = new LabelCount();
                r.label = a.label;
                r.count = a.count + b.count;
                r.mass = a.mass + b.mass;
                return r;
            });

        Map<Integer, Long> rows = new HashMap<>();
        Map<Integer, Double> mass = new HashMap<>();
        for (LabelCount c : FlinkJobs.collectAll(counts, "cluster-sizes")) {
            rows.put(c.label, c.count);
            mass.put(c.label, c.mass);
        }
        return new LabelStats(rows, mass);
    }

    /** Per-label row counts and weight masses, from one labelling scan. Equal maps on unweighted
     *  input; {@code mass} is what the weight-aware metrics need. */
    private static final class LabelStats {

        private final Map<Integer, Long> rows;
        private final Map<Integer, Double> mass;

        LabelStats(Map<Integer, Long> rows, Map<Integer, Double> mass) {
            this.rows = rows;
            this.mass = mass;
        }

        /** Non-noise clusters actually present in the labelling. */
        int clusterCount() {
            int n = 0;
            for (Integer label : rows.keySet()) {
                if (label >= 0) {
                    n++;
                }
            }
            return n;
        }

        /** Fraction of points labelled noise (cluster id -1). */
        double noiseFraction() {
            long total = 0L;
            for (long count : rows.values()) {
                total += count;
            }
            long noise = rows.getOrDefault(NOISE_LABEL, 0L);
            return total == 0L ? 0.0 : (double) noise / total;
        }

        /** Per-cluster mass with noise removed — the population the silhouette's draw represents. */
        Map<Integer, Double> clusteredMass() {
            Map<Integer, Double> clustered = new HashMap<>();
            for (Map.Entry<Integer, Double> entry : mass.entrySet()) {
                if (entry.getKey() >= 0) {
                    clustered.put(entry.getKey(), entry.getValue());
                }
            }
            return clustered;
        }

        /** Labelled (non-noise) ROW count — what an unsampled silhouette would collect. */
        long clusteredRows() {
            long total = 0L;
            for (Map.Entry<Integer, Long> entry : rows.entrySet()) {
                if (entry.getKey() >= 0) {
                    total += entry.getValue();
                }
            }
            return total;
        }

        /** {@code clusterId(string) -> rows}, ascending by label so the emitted JSON is stable
         *  across runs regardless of which subtask reported first. Noise is included, as it is on
         *  Spark: {@code clusterSizes} describes the whole labelling. */
        Map<String, Long> sizesByLabel() {
            Map<String, Long> sizes = new LinkedHashMap<>();
            for (Integer label : new TreeSet<>(rows.keySet())) {
                sizes.put(Integer.toString(label), rows.get(label));
            }
            return sizes;
        }
    }

    static final class LabelMap implements MapFunction<WeightedPoint, LabelCount> {

        private final Model model;

        LabelMap(Model model) {
            this.model = model;
        }

        @Override
        public LabelCount map(WeightedPoint point) {
            LabelCount c = new LabelCount();
            c.label = model.predict(point.features);
            // ROW count, not mass — both engines share this gap on purpose so their cluster-size
            // columns stay comparable. The MASS travels alongside for the weight-aware metrics.
            c.count = 1L;
            c.mass = Weights.safe(point.weight);
            return c;
        }
    }

    /** Per-label (rows, mass), shuffled (keyBy/reduce) in the distributed label-stats job.
     *  Public static so Flink's TypeExtractor treats it as a POJO (fast path, not Kryo). */
    public static final class LabelCount {

        public int label;
        public long count;
        public double mass;

        public LabelCount() {}
    }


}
