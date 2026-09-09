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

    /** Over-draw factor for Bernoulli sampling to ensure sufficient sample size. */
    public static final double OVERSAMPLE_FACTOR = 1.3;

    /** Calculates inclusion probability for a Bernoulli draw, incorporating the over-draw factor. */
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
            PointSource source, EnvFactory envFactory, double inclusionProbability, long seed,
            String jobName) {

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
                seed, getRuntimeContext().getIndexOfThisSubtask(), new double[] {inclusionProbability});
        }

        @Override
        public boolean filter(WeightedPoint point) {
            return draw.take(0);
        }
    }

    /**
     * Independent Bernoulli draws over one pass of the data, one stream per subset being drawn.
     *
     * <p><b>Skips, not coin flips.</b> A Bernoulli(p) draw over n elements needs n random numbers
     * only if it is written as one flip per element. The number of elements between two successes
     * is Geometric(p), so drawing that GAP directly gives the identical distribution for one random
     * number per SELECTED element. CLARA's sampling round used to flip {@code numSamples} coins per
     * point — 500 M {@code nextDouble()} calls on a 100 M-row source at the default 5 samples, which
     * was the entire CPU cost of that round; the same draw now costs about 6 500. The gap formula is
     * the standard inverse-CDF one, {@code floor(log u / log(1-p))}.
     *
     * <p><b>SplittableRandom, not java.util.Random.</b> {@code Random} guards its seed with an
     * {@code AtomicLong}, so every draw is a CAS even though these generators are strictly
     * subtask-local. {@code SplittableRandom} is the same idea without the atomics.
     *
     * <p>Both changes alter WHICH rows are drawn, and neither breaks anything the benchmark holds:
     * the two engines already draw different samples of the same size from the same data — Spark
     * seeds Catalyst's {@code rand(seed + s)} per partition, Flink seeds per subtask — so sample
     * identity was never a cross-engine invariant, only sample DISTRIBUTION is, and that is
     * unchanged. What is emphatically NOT touched is
     * {@link #generateRandomPermutation}: that one reproduces {@code scala.util.Random.shuffle}
     * bit-for-bit on purpose, so both engines' local solvers visit candidates in the same order.
     *
     * <p>One generator per stream, seeded from {@code (seed, streamIndex, subtaskId)}. That is not
     * a detail: a single shared stream made {@code pamae}'s candidate pool a draw off the very
     * sequence CLARA's samples came from — at {@code numSamples: 1} literally a nested superset of
     * the seeding sample rather than an independent draw, since both were
     * {@code new Random(seed + 31·subtask)} walked over the same rows in the same order.
     */
    public static final class BernoulliStreams {

        private final double[] probabilities;
        private final SplittableRandom[] generators;
        private final long[] countdown;

        /** One stream per entry of {@code probabilities}, independent of each other and of the
         *  streams of every other subtask. */
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

        /** Advances {@code stream} by one element and says whether that element is drawn into it. */
        public boolean take(int stream) {
            if (countdown[stream] > 0L) {
                countdown[stream]--;
                return false;
            }
            countdown[stream] = nextSkip(generators[stream], probabilities[stream]);
            return true;
        }

        /** Elements to skip before the next success of a Bernoulli(p) stream. */
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
