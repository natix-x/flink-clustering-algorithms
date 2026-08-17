package clustering.algorithms.dbscan;

import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.PointSource;
import clustering.core.Points;
import clustering.distance.DistanceMetric;
import org.apache.flink.api.common.functions.RichFilterFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/** Uniform random subsample — the paper's O(n) strategy.
 *
 *  Fraction is {@code (m + 3*sqrt(m))/n}, not {@code m/n}: the filter is a per-row Bernoulli
 *  trial, so {@code m/n} returns {@code m +- sqrt(m)} rows and half the seeds undershoot. The
 *  overshoot is trimmed to exactly m on the driver with a seeded shuffle — not by taking a
 *  prefix, which would bias towards the first subtasks (and towards the first rows of data
 *  ordered by position: Gaia, TLC).
 *
 *  Determinism caveat, as everywhere else in this repo: each subtask seeds its RNG from the
 *  run seed and its own subtask id, so the drawn set is reproducible for a FIXED parallelism. */
public final class UniformSelection implements CandidateSelectionStrategy {

    public static final UniformSelection INSTANCE = new UniformSelection();

    private UniformSelection() {}

    @Override
    public String strategyName() {
        return "uniform";
    }

    @Override
    public double[][] selectCandidates(PointSource source, EnvFactory envs, long totalRowCount,
                                       int targetCandidateCount, long seed, DistanceMetric distanceMetric) {
        if (targetCandidateCount >= totalRowCount) {
            // s = 1.0: the exactness oracle — every point is a candidate.
            return Points.toArray(Datasets.collectAll(source, envs));
        }
        double oversampledTarget = targetCandidateCount + 3.0 * Math.sqrt(targetCandidateCount);
        double fraction = Math.min(1.0, oversampledTarget / totalRowCount);

        StreamExecutionEnvironment env = envs.newEnv();
        DataStream<DenseVector> sampled = source.create(env).filter(new BernoulliFilter(fraction, seed));
        List<double[]> collected = Points.toList(FlinkJobs.collectAll(sampled, "dbscanpp-uniform-candidates"));

        if (collected.size() <= targetCandidateCount) {
            return collected.toArray(new double[0][]);
        }
        List<Integer> order = new ArrayList<>(collected.size());
        for (int i = 0; i < collected.size(); i++) {
            order.add(i);
        }
        Collections.shuffle(order, new Random(seed));
        List<Integer> kept = new ArrayList<>(order.subList(0, targetCandidateCount));
        Collections.sort(kept);
        double[][] out = new double[targetCandidateCount][];
        for (int i = 0; i < targetCandidateCount; i++) {
            out[i] = collected.get(kept.get(i));
        }
        return out;
    }

    /** Seeded per-subtask Bernoulli filter, same shape as the one CLARA samples with. */
    static final class BernoulliFilter extends RichFilterFunction<DenseVector> {
        private final double fraction;
        private final long seed;
        private transient Random rng;

        BernoulliFilter(double fraction, long seed) {
            this.fraction = fraction;
            this.seed = seed;
        }

        @Override
        public void open(Configuration parameters) {
            rng = new Random(seed + 31L * getRuntimeContext().getIndexOfThisSubtask());
        }

        @Override
        public boolean filter(DenseVector value) {
            return rng.nextDouble() < fraction;
        }
    }
}
