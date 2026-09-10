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

/**
 * Computes post-fit evaluation metrics.
 * Ensures metric definitions and handling of edge cases are consistent across runs.
 */
public final class EvaluationRunner {

    /** Supported evaluation metric names. */
    public static final Set<String> KNOWN_METRICS = new HashSet<>(Arrays.asList(
        "silhouette", "nClusters", "clusterSizes", "noiseFraction", "daviesBouldin", "calinskiHarabasz"
    ));

    /**
     * Maximum number of labeled rows allowed for an unsampled silhouette evaluation
     * to prevent driver OutOfMemory errors and excessive O(n^2) computational cost.
     */
    public static final long FULL_SILHOUETTE_MAX_ROWS = 50_000L;

    private static final int NOISE_LABEL = -1;

    private final DistanceMetric distance;

    public EvaluationRunner(DistanceMetric distance) {
        this.distance = distance;
    }

    public EvaluationResult run(
            Model model,
            PointSource source,
            EnvFactory envs,
            EvaluationSpec spec,
            int parallelism
    ) {
        Set<String> metrics = new HashSet<>(spec.metrics == null ? Collections.emptyList() : spec.metrics);
        if (!KNOWN_METRICS.containsAll(metrics)) {
            Set<String> unknown = new TreeSet<>(metrics);
            unknown.removeAll(KNOWN_METRICS);
            throw new IllegalArgumentException(
                "unknown evaluation metric(s): " + String.join(", ", unknown) +
                "; known: " + String.join(", ", new TreeSet<>(KNOWN_METRICS))
            );
        }

        boolean wantSilhouette = metrics.contains("silhouette");
        boolean wantSizes = metrics.contains("clusterSizes");
        boolean wantNoise = metrics.contains("noiseFraction");
        boolean wantNClusters = metrics.contains("nClusters");
        boolean wantDaviesBouldin = metrics.contains("daviesBouldin");
        boolean wantCalinskiHarabasz = metrics.contains("calinskiHarabasz");

        LabelStats stats = (wantSizes || wantNoise || wantNClusters || wantSilhouette)
            ? computeStats(model, source, envs) : null;

        ClusterMoments moments = (wantDaviesBouldin || wantCalinskiHarabasz)
            ? ClusterMoments.compute(model, source, envs, distance) : null;

        SilhouetteEvaluator.Outcome silhouette = wantSilhouette
            ? computeSilhouette(model, source, envs, spec, parallelism, stats) : null;

        return new EvaluationResult(
            stats == null ? null : stats.clusterCount(),
            wantNoise && stats != null ? stats.noiseFraction() : null,
            silhouette == null ? null : silhouette.score,
            silhouette == null ? null : silhouette.scoredPoints,
            silhouette == null ? null : silhouette.sampleClusters,
            silhouette == null ? null : silhouette.unscoredPoints,
            wantSizes && stats != null ? stats.sizesByLabel() : null,
            wantDaviesBouldin && moments != null ? DaviesBouldinIndex.compute(moments, distance) : null,
            wantCalinskiHarabasz && moments != null ? CalinskiHarabaszIndex.compute(moments, distance) : null
        );
    }

    /**
     * Computes the silhouette score on the full dataset or a subsample.
     * Enforces the FULL_SILHOUETTE_MAX_ROWS limit for unsampled evaluations.
     */
    private SilhouetteEvaluator.Outcome computeSilhouette(
            Model model,
            PointSource source,
            EnvFactory envs,
            EvaluationSpec spec,
            int parallelism,
            LabelStats stats
    ) {
        SilhouetteEvaluator.Population population =
            stats == null ? null : new SilhouetteEvaluator.Population(stats.clusteredMass());

        long labelledRows = stats == null ? Long.MAX_VALUE : stats.clusteredRows();
        boolean sampled = spec.sampleSize() != null && spec.sampleSize() > 0;

        if (!sampled && labelledRows > FULL_SILHOUETTE_MAX_ROWS) {
            throw new IllegalArgumentException(
                "silhouette requested with no sampleSize on " + labelledRows +
                " labelled rows, above the " + FULL_SILHOUETTE_MAX_ROWS +
                " the driver can collect. Set evaluation.sampleSize or run on a smaller dataset."
            );
        }

        return new SilhouetteEvaluator(distance)
            .measure(model, source, envs, parallelism, spec.sampleSize(), spec.seed, population);
    }

    /**
     * Computes cluster row counts and weight masses in a single distributed scan.
     */
    private LabelStats computeStats(Model model, PointSource source, EnvFactory envs) {
        StreamExecutionEnvironment env = envs.newEnv();
        DataStream<LabelCount> counts = source.create(env)
            .map(new LabelMap(model))
            .returns(LabelCount.class)
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

    /**
     * Encapsulates per-label row counts and weight masses.
     */
    private static final class LabelStats {

        private final Map<Integer, Long> rows;
        private final Map<Integer, Double> mass;

        LabelStats(Map<Integer, Long> rows, Map<Integer, Double> mass) {
            this.rows = rows;
            this.mass = mass;
        }

        int clusterCount() {
            int n = 0;
            for (Integer label : rows.keySet()) {
                if (label >= 0) {
                    n++;
                }
            }
            return n;
        }

        double noiseFraction() {
            long total = 0L;
            for (long count : rows.values()) {
                total += count;
            }
            long noise = rows.getOrDefault(NOISE_LABEL, 0L);
            return total == 0L ? 0.0 : (double) noise / total;
        }

        Map<Integer, Double> clusteredMass() {
            Map<Integer, Double> clustered = new HashMap<>();
            for (Map.Entry<Integer, Double> entry : mass.entrySet()) {
                if (entry.getKey() >= 0) {
                    clustered.put(entry.getKey(), entry.getValue());
                }
            }
            return clustered;
        }

        long clusteredRows() {
            long total = 0L;
            for (Map.Entry<Integer, Long> entry : rows.entrySet()) {
                if (entry.getKey() >= 0) {
                    total += entry.getValue();
                }
            }
            return total;
        }

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
            c.count = 1L;
            c.mass = Weights.sanitize(point.weight);
            return c;
        }
    }

    /**
     * POJO for accumulating row counts and weight masses per cluster.
     */
    public static final class LabelCount {
        public int label;
        public long count;
        public double mass;

        public LabelCount() {}
    }
}
