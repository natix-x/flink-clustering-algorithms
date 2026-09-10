package clustering.algorithms.dbscan.components;

import clustering.core.WeightedPoint;
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

/**
 * Uniform random subsampling for DBSCAN++ candidate core points.
 * Uses oversampled Bernoulli trials followed by exact truncation to achieve the target sample size.
 *
 * <p>This used to sit behind a {@code CandidateSelectionStrategy} interface, but uniform is the
 * only selection ever implemented — the interface was pure indirection. If a second strategy
 * shows up, re-extract the interface then.
 */
public final class UniformSelection {

    public static final UniformSelection INSTANCE = new UniformSelection();

    private UniformSelection() {}

    /** Resolves a {@code sampling} param value. */
    public static UniformSelection fromName(String name) {
        switch (name.toLowerCase()) {
            case "uniform":
                return INSTANCE;
            default:
                throw new IllegalArgumentException(
                    "Unknown candidate sampling: '" + name + "'. Known: uniform"
                );
        }
    }

    public String strategyName() {
        return "uniform";
    }

    /**
     * Selects at most {@code targetCount} candidates from the source.
     */
    public double[][] selectCandidates(PointSource source, EnvFactory envFactory, long totalPoints,
                                       int targetCount, long seed, DistanceMetric distanceMetric) {
        if (targetCount >= totalPoints) {
            return Points.toArray(Datasets.collectAll(source, envFactory));
        }

        double adjustedTarget = targetCount + 3.0 * Math.sqrt(targetCount);
        double samplingFraction = Math.min(1.0, adjustedTarget / totalPoints);

        StreamExecutionEnvironment env = envFactory.newEnv();
        DataStream<WeightedPoint> sampledStream = source.create(env).filter(new BernoulliFilter(samplingFraction, seed));
        List<double[]> drawnPoints = Points.toList(FlinkJobs.collectAll(sampledStream, "dbscanpp-uniform-candidates"));

        if (drawnPoints.size() <= targetCount) {
            return drawnPoints.toArray(new double[0][]);
        }

        List<Integer> indices = new ArrayList<>(drawnPoints.size());
        for (int i = 0; i < drawnPoints.size(); i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(seed));

        List<Integer> selectedIndices = new ArrayList<>(indices.subList(0, targetCount));
        Collections.sort(selectedIndices); // Preserve original relative stream order

        double[][] finalCandidates = new double[targetCount][];
        for (int i = 0; i < targetCount; i++) {
            finalCandidates[i] = drawnPoints.get(selectedIndices.get(i));
        }
        return finalCandidates;
    }

    /** Stateful Bernoulli filter for distributed subtask sampling. */
    static final class BernoulliFilter extends RichFilterFunction<WeightedPoint> {
        private final double samplingFraction;
        private final long seed;
        private transient Random randomGenerator;

        BernoulliFilter(double samplingFraction, long seed) {
            this.samplingFraction = samplingFraction;
            this.seed = seed;
        }

        @Override
        public void open(Configuration parameters) {
            randomGenerator = new Random(seed + 31L * getRuntimeContext().getIndexOfThisSubtask());
        }

        @Override
        public boolean filter(WeightedPoint point) {
            return randomGenerator.nextDouble() < samplingFraction;
        }
    }
}
