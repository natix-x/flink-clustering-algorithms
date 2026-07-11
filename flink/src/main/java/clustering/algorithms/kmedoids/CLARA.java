package clustering.algorithms.kmedoids;

import clustering.core.Clusterer;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import org.apache.flink.api.common.functions.RichFilterFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.List;
import java.util.Random;

/** CLARA (Clustering LARge Applications) — scalable k-medoids. Java mirror of the Spark
 *  {@code CLARA}.
 *
 *  PAM is O(n²); CLARA makes it scale by drawing several small random samples from the
 *  full (distributed) dataset, running PAM on each sample on the driver, and keeping the
 *  medoid set whose assignment cost — measured over the FULL dataset — is lowest. Only
 *  the per-sample PAM and the cost reduction touch Flink:
 *    - sampling   : one Flink job per sample (seeded fraction filter, collected to driver),
 *    - PAM        : driver-local on the small sample via {@link PAM#fitLocal},
 *    - cost eval  : one distributed Flink job per sample (sum of nearest-medoid distances). */
public class CLARA implements Clusterer {

    private final int k;
    private final int numSamples;
    private final int sampleSize;
    private final DistanceMetric distance;
    private final PAM pam;

    public CLARA(int k, int numSamples, int sampleSize, int maxIter, DistanceMetric distance) {
        this.k = k;
        this.numSamples = numSamples;
        this.sampleSize = sampleSize;
        this.distance = distance;
        this.pam = new PAM(k, maxIter, distance);
    }

    @Override
    public Model fit(PointSource source, EnvFactory envs, int parallelism) {
        long n = Datasets.count(source, envs);

        KMedoidsModel bestModel = null;
        double minCost = Double.MAX_VALUE;

        for (int i = 0; i < numSamples; i++) {
            // Phase 1: sample from the full distributed dataset (oversample 2x, then cap).
            double fraction = Math.min(1.0, sampleSize * 2.0 / n);
            double[][] sample = collectSample(source, envs, fraction, i);

            if (sample.length < k) {
                throw new IllegalArgumentException(
                    "Sample too small: got " + sample.length + " points, need at least k=" + k + ".");
            }

            // Phase 2: run PAM locally on the small sample (O(sampleSize²)).
            KMedoidsModel candidate = pam.fitLocal(sample);

            // Phase 3: evaluate candidate quality on the full distributed dataset.
            double cost = evaluateCost(source, envs, candidate.medoids());

            if (cost < minCost) {
                minCost = cost;
                bestModel = candidate;
            }
        }

        return bestModel;
    }

    /** One Flink job: keeps each point with probability {@code fraction} (seeded per
     *  sample), collects up to {@code sampleSize} to the driver. */
    private double[][] collectSample(PointSource source, EnvFactory envs, double fraction, int sampleIdx) {
        StreamExecutionEnvironment env = envs.newEnv();
        DataStream<double[]> sampled = source.create(env)
            .filter(new SampleFilter(fraction, sampleIdx));
        List<double[]> out = FlinkJobs.collectUpTo(sampled, "clara-sample-" + sampleIdx, sampleSize);
        return out.toArray(new double[0][]);
    }

    /** One distributed Flink job: total assignment cost (sum over all points of the
     *  distance to the nearest medoid). */
    private double evaluateCost(PointSource source, EnvFactory envs, double[][] medoids) {
        StreamExecutionEnvironment env = envs.newEnv();
        DataStream<Double> totals = source.create(env)
            .map(new NearestMedoidDistance(medoids, distance)).returns(Types.DOUBLE)
            .keyBy(x -> 0).reduce((a, b) -> a + b);
        Double cost = FlinkJobs.last(totals, "clara-cost");  // BATCH keyed reduce emits one final value
        return cost == null ? Double.MAX_VALUE : cost;
    }

    /** Seeded Bernoulli sample filter. Each subtask seeds its RNG from the sample index
     *  and its subtask id so samples are repeatable for a given parallelism. */
    private static final class SampleFilter extends RichFilterFunction<double[]> {
        private final double fraction;
        private final int sampleIdx;
        private transient Random rng;

        SampleFilter(double fraction, int sampleIdx) {
            this.fraction = fraction;
            this.sampleIdx = sampleIdx;
        }

        @Override
        public void open(Configuration parameters) {
            int subtask = getRuntimeContext().getIndexOfThisSubtask();
            rng = new Random(31L * (sampleIdx + 1) + subtask);
        }

        @Override
        public boolean filter(double[] value) {
            return rng.nextDouble() < fraction;
        }
    }

    /** Maps a point to its distance to the nearest medoid. */
    private static final class NearestMedoidDistance
            implements org.apache.flink.api.common.functions.MapFunction<double[], Double> {
        private final double[][] medoids;
        private final DistanceMetric distance;

        NearestMedoidDistance(double[][] medoids, DistanceMetric distance) {
            this.medoids = medoids;
            this.distance = distance;
        }

        @Override
        public Double map(double[] features) {
            double min = Double.MAX_VALUE;
            for (double[] medoid : medoids) {
                double d = distance.compute(features, medoid);
                if (d < min) {
                    min = d;
                }
            }
            return min;
        }
    }
}