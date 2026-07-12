package clustering.benchmark.evaluation;

import clustering.benchmark.config.EvaluationSpec;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import clustering.evaluation.SilhouetteEvaluator;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


public final class EvaluationRunner {

    /** Default silhouette sample cap when {@code EvaluationSpec.sampleSize} is unset. */
    private static final long DEFAULT_SILHOUETTE_CAP = 10_000L;

    private final DistanceMetric distance;

    public EvaluationRunner(DistanceMetric distance) {
        this.distance = distance;
    }

    public EvaluationResult run(Model model, PointSource source, EnvFactory envs,
                                EvaluationSpec spec, long nRows) {
        boolean wantSizes = spec.wants("clusterSizes");
        boolean wantNoise = spec.wants("noiseFraction");
        boolean wantNClusters = spec.wants("nClusters");

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
        Double silhouette = spec.wants("silhouette")
            ? computeSilhouette(model, source, envs, spec) : null;

        return new EvaluationResult(nClusters, noiseFraction, silhouette, wantSizes ? sizes : null);
    }

    /** Silhouette on a driver-side head sample (uniform first-by-index; O(n^2) evaluator). */
    private double computeSilhouette(Model model, PointSource source, EnvFactory envs, EvaluationSpec spec) {
        long cap = spec.sampleSize != null ? spec.sampleSize : DEFAULT_SILHOUETTE_CAP;
        List<double[]> sample = Datasets.collectHead(source, envs, cap);
        return new SilhouetteEvaluator(distance).evaluate(sample, model.labels(sample));
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
        for (LabelCount c : FlinkJobs.collectAll(counts, "cluster-sizes")) {
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

    /** Per-label count, shuffled (keyBy/reduce) in the distributed cluster-sizes job.
     *  Public static so Flink's TypeExtractor treats it as a POJO (fast path, not Kryo). */
    public static final class LabelCount {
        public int label;
        public long count;

        public LabelCount() {}
    }
}
