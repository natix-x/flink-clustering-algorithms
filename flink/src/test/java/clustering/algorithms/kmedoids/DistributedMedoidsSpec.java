package clustering.algorithms.kmedoids;

import clustering.algorithms.kmedoids.components.MedoidRefinement;
import clustering.algorithms.kmedoids.hybrid.CLARA;
import clustering.algorithms.kmedoids.hybrid.PAMAE;
import clustering.TestFixtures;
import clustering.core.EnvFactory;
import clustering.core.PointSource;
import clustering.distance.EuclideanDistance;
import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.WeightedPoint;
import clustering.core.WeightedPointTypeInfo;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The distributed end of the k-medoids ladder: {@code pamae} (seeding + refinement), the shared
 *  refinement primitive and the {@code inner} knob on CLARA. Mirrors the Spark
 *  {@code DistributedMedoidsSpec}.
 *
 *  The load-bearing assertion is {@code pamae <= clara}: PAMAE is CLARA plus one refinement phase
 *  over the entire data, so if refinement is implemented correctly the objective can only improve.
 *  That is also the number the thesis reports as "what the entire-data half buys". */
class DistributedMedoidsSpec {

    private static final int PARALLELISM = 2;

    /** Four Gaussian blobs, 250 points each, fixed seed — enough structure that a bad medoid set
     *  costs visibly more than a good one. */
    private static List<double[]> fourBlobs() {
        double[][] centres = {{0, 0}, {30, 0}, {0, 30}, {30, 30}};
        Random rng = new Random(11L);
        List<double[]> rows = new ArrayList<>();
        for (double[] c : centres) {
            for (int i = 0; i < 250; i++) {
                rows.add(new double[] {c[0] + rng.nextGaussian() * 2, c[1] + rng.nextGaussian() * 2});
            }
        }
        return rows;
    }

    /** 20 000 rows, x strictly increasing, CONTIGUOUS ranges per subtask — the shape a parquet
     *  split has, and the one a positional bias bites on (see {@link ClaraSamplingSpec}). */
    private static PointSource orderedSource(int rows, int parallelism) {
        return env -> env.fromSequence(0, rows - 1)
            .setParallelism(parallelism)
            .map(i -> WeightedPoint.of(new double[] {i, 0.0}))
            .returns(WeightedPointTypeInfo.INSTANCE)
            .setParallelism(parallelism);
    }

    private static double cost(List<double[]> rows, DenseVector[] medoids) {
        double total = 0.0;
        for (double[] x : rows) {
            double min = Double.MAX_VALUE;
            for (DenseVector m : medoids) {
                min = Math.min(min, EuclideanDistance.INSTANCE.compute(x, m.values));
            }
            total += min;
        }
        return total;
    }

    /** The refinement primitive on its own: started from a deliberately bad medoid set, it must
     *  lower the objective and still return k medoids that are real data points. */
    @Test
    void refinementImprovesOnItsInitialisationAndReturnsKDataPoints() {
        List<double[]> rows = fourBlobs();
        PointSource source = TestFixtures.source(rows);
        EnvFactory envs = TestFixtures.localEnvs(PARALLELISM);

        DenseVector[] pool = MedoidRefinement.sampleCandidatePool(source, envs, rows.size(), 200, 5L);
        assertTrue(pool.length >= 4, "pool must hold at least k candidates, got " + pool.length);
        DenseVector[] initial = {pool[0], pool[1], pool[2], pool[3]};

        MedoidRefinement.Result refined = MedoidRefinement.refine(
            source, envs, initial, pool, EuclideanDistance.INSTANCE, 20);

        assertEquals(4, refined.medoids.length);
        Set<String> distinct = new HashSet<>();
        for (DenseVector m : refined.medoids) {
            distinct.add(java.util.Arrays.toString(m.values));
        }
        assertEquals(4, distinct.size(), "medoids must be distinct");

        Set<String> inputs = new HashSet<>();
        for (double[] x : rows) {
            inputs.add(java.util.Arrays.toString(x));
        }
        for (DenseVector m : refined.medoids) {
            assertTrue(inputs.contains(java.util.Arrays.toString(m.values)),
                "every medoid must be a data point");
        }
        assertTrue(cost(rows, refined.medoids) < cost(rows, initial),
            "refinement did not improve: " + cost(rows, refined.medoids)
                + " vs initial " + cost(rows, initial));
    }

    /** The candidate pool of phase II is the set it may pick representatives from, so a pool drawn
     *  from one region of an ordered input silently restricts the very phase that exists to see ALL
     *  the data. Same defect shape as the CLARA sampling test. */
    @Test
    void pamaesCandidatePoolIsDrawnFromTheWholeDatasetNotAPrefix() {
        int rows = 20_000;
        for (int parallelism : new int[] {1, PARALLELISM}) {
            PointSource source = orderedSource(rows, parallelism);
            EnvFactory envs = TestFixtures.localEnvs(parallelism);
            DenseVector[] pool =
                MedoidRefinement.sampleCandidatePool(source, envs, rows, 400, 23L);

            assertEquals(400, pool.length, "expected a full pool");
            double max = 0.0;
            double sum = 0.0;
            for (DenseVector v : pool) {
                max = Math.max(max, v.values[0]);
                sum += v.values[0];
            }
            double mean = sum / pool.length;
            assertTrue(mean > 8_000.0 && mean < 12_000.0,
                "pool mean " + mean + " is not centred — draw is biased (p=" + parallelism + ")");
            assertTrue(max > 18_000.0,
                "pool never reaches the tail of the data (max " + max + ", p=" + parallelism + ")");
        }
    }

    @Test
    void pamaeNeverScoresWorseThanItsOwnSeedingPhaseClara() {
        List<double[]> rows = fourBlobs();
        PointSource source = TestFixtures.source(rows);
        EnvFactory envs = TestFixtures.localEnvs(PARALLELISM);

        KMedoidsModel clara = (KMedoidsModel) new CLARA(
            4, 3, 120, 50, EuclideanDistance.INSTANCE, "fastpam", 9L)
            .fit(source, envs, PARALLELISM);
        KMedoidsModel pamae = (KMedoidsModel) new PAMAE(
            4, 3, 120, 50, 3, 200, EuclideanDistance.INSTANCE, "fastpam", 9L)
            .fit(source, envs, PARALLELISM);

        double claraCost = cost(rows, clara.medoids());
        double pamaeCost = cost(rows, pamae.medoids());
        assertTrue(pamaeCost <= claraCost + 1e-9,
            "refinement made the objective worse: pamae=" + pamaeCost + " clara=" + claraCost);
    }

    /** Monotonicity is a guarantee of {@link MedoidRefinement#refine} — incumbents are always
     *  candidates and a slot only moves on a STRICT improvement — so it is tested where it holds:
     *  the SAME pool and the SAME initial medoids, varying only the iteration count.
     *
     *  <p>Comparing two full {@code pamae} runs with different {@code refineIters} does NOT test
     *  this, and the earlier version of this test which did so was wrong. The candidate pool is a
     *  random draw whose contents vary between runs (Flink's rebalance partitioner starts at a
     *  random channel), so the two runs refine against different pools and the comparison measures
     *  the pool difference as much as the knob. It duly failed: 2546.22 at 5 iterations against
     *  2531.71 at 1. Pinning the draw to make that comparison work was tried and reverted — the
     *  variance is CLARA's and PAMAE's property, and the fix belongs in the test, which can hold the
     *  pool fixed because {@code refine} takes it as a parameter. */
    @Test
    void refinementIsMonotoneInItsIterationCount() {
        List<double[]> rows = fourBlobs();
        PointSource source = TestFixtures.source(rows);
        EnvFactory envs = TestFixtures.localEnvs(PARALLELISM);

        DenseVector[] pool = MedoidRefinement.sampleCandidatePool(source, envs, rows.size(), 150, 4L);
        DenseVector[] initial = {pool[0], pool[1], pool[2], pool[3]};

        double previous = Double.MAX_VALUE;
        for (int iterations : new int[] {1, 2, 3, 5, 8}) {
            MedoidRefinement.Result result = MedoidRefinement.refine(
                source, envs, initial, pool, EuclideanDistance.INSTANCE, iterations);
            double cost = cost(rows, result.medoids);
            assertTrue(cost <= previous + 1e-9,
                "refining for " + iterations + " iterations cost " + cost
                    + ", more than the shorter run's " + previous);
            previous = cost;
        }
    }

    /** What a {@code refineIters} sweep on the full entry can be asserted to do: never make the
     *  seeding worse. Each run draws its own pool, so the costs are not comparable to each other —
     *  but every one of them is still refinement applied on top of CLARA, and that is monotone by
     *  construction whatever pool it got. */
    @Test
    void everyRefineItersSettingImprovesOnTheSeedingItStartedFrom() {
        List<double[]> rows = fourBlobs();
        PointSource source = TestFixtures.source(rows);
        EnvFactory envs = TestFixtures.localEnvs(PARALLELISM);

        KMedoidsModel clara = (KMedoidsModel) new CLARA(
            4, 2, 80, 50, EuclideanDistance.INSTANCE, "fastpam", 4L)
            .fit(source, envs, PARALLELISM);
        double seedingCost = cost(rows, clara.medoids());

        for (int refineIters : new int[] {1, 5}) {
            KMedoidsModel pamae = (KMedoidsModel) new PAMAE(
                4, 2, 80, 50, refineIters, 150, EuclideanDistance.INSTANCE, "fastpam", 4L)
                .fit(source, envs, PARALLELISM);
            double refinedCost = cost(rows, pamae.medoids());
            assertTrue(refinedCost <= seedingCost + 1e-9,
                "refineIters=" + refineIters + " scored " + refinedCost
                    + ", worse than the seeding's " + seedingCost);
        }
    }

    /* CLARA's batched scoring (all candidate sets in ONE pass) used to be a standalone
     * `MedoidCost` helper with its own batched-equals-one-by-one test. Both are gone: the fold now
     * lives inside CLARA's iteration, where it costs no extra job, and a scrambled candidate index
     * in it shows up as a wrong winner — which the well-separated-blob assertions below and
     * `pamae <= clara` already catch, since a wrong pick means a worse objective. */

    @Test
    void claraInnerSolverIsAKnobAndBothSolversAgreeOnWellSeparatedBlobs() {
        List<double[]> rows = fourBlobs();
        PointSource source = TestFixtures.source(rows);
        EnvFactory envs = TestFixtures.localEnvs(PARALLELISM);

        double best = Double.MAX_VALUE;
        double[] costs = new double[2];
        String[] inners = {"fastpam", "fasterpam"};
        for (int i = 0; i < inners.length; i++) {
            KMedoidsModel model = (KMedoidsModel) new CLARA(
                4, 3, 120, 50, EuclideanDistance.INSTANCE, inners[i], 9L)
                .fit(source, envs, PARALLELISM);
            costs[i] = cost(rows, model.medoids());
            best = Math.min(best, costs[i]);
        }
        // Same objective, two searches: on separated blobs they must land within a few percent.
        for (int i = 0; i < inners.length; i++) {
            assertTrue(costs[i] <= best * 1.05,
                "inner='" + inners[i] + "' cost " + costs[i] + " is more than 5% worse than " + best);
        }
    }

    @Test
    void claraRejectsAnUnknownInnerSolver() {
        assertTrue(assertThrows(IllegalArgumentException.class,
            () -> new CLARA(2, 5, 100, 50, EuclideanDistance.INSTANCE, "banditpam", 42L))
            .getMessage().contains("Unknown inner k-medoids solver"));
    }

    @Test
    void pamaeRejectsZeroRefinementIterations() {
        assertTrue(assertThrows(IllegalArgumentException.class,
            () -> new PAMAE(2, 5, 100, 50, 0, 100, EuclideanDistance.INSTANCE, "fastpam", 42L))
            .getMessage().contains("refineIters"));
    }
}
