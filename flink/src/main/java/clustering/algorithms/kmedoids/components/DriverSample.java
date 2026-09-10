package clustering.algorithms.kmedoids.components;

import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.PointSource;
import clustering.core.WeightedPoint;
import org.apache.flink.api.common.functions.RichFilterFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;

/**
 * Utility for drawing random data subsets to the driver.
 * Ensures uniform sampling without scan-order or positional biases.
 */
public final class DriverSample {

    private DriverSample() {}

    public static final double OVERSAMPLE_FACTOR = 1.3;

    /**
     * Calculates inclusion probability for a Bernoulli draw, incorporating the over-draw factor.
     */
    public static double calculateInclusionProbability(int targetSize, long totalRowCount) {
        return Math.min(1.0, OVERSAMPLE_FACTOR * targetSize / (double) totalRowCount);
    }

    /**
     * Selects a uniform random subset of the specified size.
     * Shuffles the list before truncating to avoid ordering bias.
     */
    public static double[][] takeRandom(List<double[]> dataset, int subsetSize, long seed) {
        if (dataset.size() <= subsetSize) {
            return dataset.toArray(new double[0][]);
        }
        int[] shuffledIndices = generateRandomPermutation(dataset.size(), seed);
        double[][] subset = new double[subsetSize][];
        for (int i = 0; i < subsetSize; i++) {
            subset[i] = dataset.get(shuffledIndices[i]);
        }
        return subset;
    }

    /**
     * Executes a Flink job to perform a distributed Bernoulli draw.
     * The resulting set size is unbounded and should be truncated by the caller.
     */
    public static List<double[]> executeBernoulliSample(
            PointSource source,
            EnvFactory envFactory,
            double inclusionProbability,
            long seed,
            String jobName
    ) {
        StreamExecutionEnvironment env = envFactory.newEnv();
        DataStream<WeightedPoint> sampledStream = source.create(env)
            .filter(new BernoulliFilter(inclusionProbability, seed));

        List<double[]> sampledCoordinates = new ArrayList<>();
        FlinkJobs.consume(sampledStream, jobName, point -> sampledCoordinates.add(point.features.values));
        return sampledCoordinates;
    }

    private static final class BernoulliFilter extends RichFilterFunction<WeightedPoint> {
        private final double inclusionProbability;
        private final long seed;
        private transient BernoulliStreams draw;

        BernoulliFilter(double inclusionProbability, long seed) {
            this.inclusionProbability = inclusionProbability;
            this.seed = seed;
        }

        @Override
        public void open(Configuration parameters) {
            draw = new BernoulliStreams(
                seed, getRuntimeContext().getIndexOfThisSubtask(), new double[] {inclusionProbability}
            );
        }

        @Override
        public boolean filter(WeightedPoint point) {
            return draw.take(0);
        }
    }

    /**
     * Independent Bernoulli draws over one pass of the data.
     * Uses geometric distribution to calculate gaps between drawn elements for performance,
     * avoiding the cost of evaluating a random coin flip for every single row.
     */
    public static final class BernoulliStreams {

        private final double[] probabilities;
        private final SplittableRandom[] generators;
        private final long[] countdown;

        public BernoulliStreams(long seed, int subtaskId, double[] probabilities) {
            this.probabilities = probabilities.clone();
            this.generators = new SplittableRandom[probabilities.length];
            this.countdown = new long[probabilities.length];

            for (int stream = 0; stream < probabilities.length; stream++) {
                generators[stream] = new SplittableRandom(streamSeed(seed, stream, subtaskId));
                countdown[stream] = nextSkip(generators[stream], this.probabilities[stream]);
            }
        }

        public int streamCount() {
            return probabilities.length;
        }

        /**
         * Advances the stream by one element and returns true if it should be drawn.
         */
        public boolean take(int stream) {
            if (countdown[stream] > 0L) {
                countdown[stream]--;
                return false;
            }
            countdown[stream] = nextSkip(generators[stream], probabilities[stream]);
            return true;
        }

        private static long nextSkip(SplittableRandom rng, double probability) {
            if (probability >= 1.0) {
                return 0L;
            }
            if (probability <= 0.0) {
                return Long.MAX_VALUE;
            }
            double uniform = rng.nextDouble();
            if (uniform <= 0.0) {
                uniform = Double.MIN_VALUE;
            }
            double skip = Math.floor(Math.log(uniform) / Math.log1p(-probability));
            return skip >= (double) Long.MAX_VALUE ? Long.MAX_VALUE : (long) skip;
        }

        private static long streamSeed(long seed, int stream, int subtaskId) {
            long mixed = seed * 0x9E3779B97F4A7C15L;
            mixed ^= (stream + 1L) * 0xBF58476D1CE4E5B9L;
            mixed ^= (subtaskId + 1L) * 0x94D049BB133111EBL;
            return mixed;
        }
    }

    /**
     * Generates a Fisher-Yates permutation.
     * Implemented manually to maintain exact reproducibility with Scala's Random.shuffle.
     */
    public static int[] generateRandomPermutation(int size, long seed) {
        int[] permutation = new int[size];
        for (int i = 0; i < size; i++) {
            permutation[i] = i;
        }
        Random rng = new Random(seed);
        for (int n = size; n > 1; n--) {
            int swapIndex = rng.nextInt(n);
            int temp = permutation[n - 1];
            permutation[n - 1] = permutation[swapIndex];
            permutation[swapIndex] = temp;
        }
        return permutation;
    }
}
