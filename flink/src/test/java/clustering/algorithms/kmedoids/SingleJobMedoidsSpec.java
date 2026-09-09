package clustering.algorithms.kmedoids;

import clustering.TestFixtures;
import clustering.algorithms.kmedoids.components.DriverSample;
import clustering.algorithms.kmedoids.distributed.DistributedFastPAM;
import clustering.algorithms.kmedoids.hybrid.CLARA;
import clustering.algorithms.kmedoids.hybrid.PAMAE;
import clustering.core.EnvFactory;
import clustering.core.PointSource;
import clustering.core.WeightedPoint;
import clustering.core.WeightedPointTypeInfo;
import clustering.distance.EuclideanDistance;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** How many Flink JOBS a distributed medoid entry costs, and the two mechanisms that made the
 *  count drop.
 *
 *  <p>The count is the measurement, not an implementation detail. Flink has no cross-job cache, so
 *  a second job is a second read of the source from storage plus a second deployment (4 934 ms of
 *  fixed cost, measured on Ares) plus a second fill of the point cache. {@code pamae} used to cost
 *  THREE — CLARA, the candidate-pool draw, the refinement — i.e. it read Cohere's 196 GB three
 *  times to do one fit. Nothing in the algorithms' output says whether that regressed, so it is
 *  asserted directly: an {@link EnvFactory} hands out exactly one environment per job.
 *
 *  <p>Sources carry their row count, as the benchmark job's does — otherwise {@code Datasets.count}
 *  is a job of its own and the numbers below would all be one higher, for a number the caller
 *  already knows. */
class SingleJobMedoidsSpec {

    private static final int PARALLELISM = 2;

    /** Counts environments handed out; one environment is one submitted job. */
    private static final class JobCountingEnvs implements EnvFactory {
        private final EnvFactory delegate;
        private int jobs;

        JobCountingEnvs(EnvFactory delegate) {
            this.delegate = delegate;
        }

        @Override
        public StreamExecutionEnvironment newEnv() {
            jobs++;
            return delegate.newEnv();
        }
    }

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

    /** Contiguous, deterministic splits — a parquet reader's shape, and the only one under which a
     *  run is reproducible at all (a {@code fromCollection} source is rebalanced starting at a
     *  RANDOM channel, so the points land in a different subtask every run). */
    private static PointSource orderedSource(int rows, int parallelism) {
        return env -> env.fromSequence(0, rows - 1)
            .setParallelism(parallelism)
            .map(i -> WeightedPoint.of(new double[] {i, 0.0}))
            .returns(WeightedPointTypeInfo.INSTANCE)
            .setParallelism(parallelism);
    }

    @Test
    void claraRunsAsASingleFlinkJob() {
        List<double[]> rows = fourBlobs();
        PointSource source = PointSource.withKnownRowCount(TestFixtures.source(rows), rows.size());
        JobCountingEnvs envs = new JobCountingEnvs(TestFixtures.localEnvs(PARALLELISM));

        new CLARA(4, 3, 120, 50, EuclideanDistance.INSTANCE, "fastpam", 9L)
            .fit(source, envs, PARALLELISM);

        assertEquals(1, envs.jobs, "clara must read the source exactly once");
    }

    /** The one that regressed: seeding, candidate pool and refinement are one phase machine now. */
    @Test
    void pamaeRunsAsASingleFlinkJobIncludingItsCandidatePool() {
        List<double[]> rows = fourBlobs();
        PointSource source = PointSource.withKnownRowCount(TestFixtures.source(rows), rows.size());
        JobCountingEnvs envs = new JobCountingEnvs(TestFixtures.localEnvs(PARALLELISM));

        new PAMAE(4, 3, 120, 50, 3, 200, EuclideanDistance.INSTANCE, "fastpam", 9L)
            .fit(source, envs, PARALLELISM);

        assertEquals(1, envs.jobs,
            "pamae must not pay a job for its candidate pool or its refinement");
    }

    /** Its candidates used to come from a {@code collectAll} at parallelism 1 before the job. */
    @Test
    void distfastpamRunsAsASingleFlinkJob() {
        List<double[]> rows = fourBlobs();
        PointSource source = PointSource.withKnownRowCount(TestFixtures.source(rows), rows.size());
        JobCountingEnvs envs = new JobCountingEnvs(TestFixtures.localEnvs(PARALLELISM));

        new DistributedFastPAM(4, 30, EuclideanDistance.INSTANCE).fit(source, envs, PARALLELISM);

        assertEquals(1, envs.jobs, "distfastpam must gather its candidates inside its own job");
    }

    /** The pre-fold that keeps {@code distfastpam}'s {@code n·(k+1)}-double partials off a single
     *  task must not cost reproducibility: groups are subtask RANGES and a group is never split, so
     *  the fold order is still fixed by slot index. Run at a parallelism ABOVE the fan-in, so more
     *  than one group actually exists. */
    @Test
    void distfastpamIsReproducibleAcrossTheMergeStage() {
        int parallelism = 16;   // > MERGE_FAN_IN, so the merge stage produces several groups
        PointSource source = orderedSource(1_000, parallelism);
        EnvFactory envs = TestFixtures.localEnvs(parallelism);

        DenseVector[] first = ((KMedoidsModel) new DistributedFastPAM(3, 20, EuclideanDistance.INSTANCE)
            .fit(source, envs, parallelism)).medoids();
        DenseVector[] second = ((KMedoidsModel) new DistributedFastPAM(3, 20, EuclideanDistance.INSTANCE)
            .fit(source, envs, parallelism)).medoids();

        assertEquals(first.length, second.length);
        for (int i = 0; i < first.length; i++) {
            assertArrayEquals(first[i].values, second[i].values, 0.0,
                "medoid " + i + " differs between two runs of one configuration");
        }
    }

    /** The draw {@code pamae}'s pool and CLARA's samples now share one pass of.
     *
     *  <p>Two properties, both of which were broken before. It must cover the WHOLE input, which a
     *  skip-based Bernoulli draw does exactly as a per-element coin flip did. And two streams must
     *  be INDEPENDENT: the pool used to be drawn from {@code new Random(seed + 31·subtask)} walked
     *  over the same rows in the same order as CLARA's samples, which at {@code numSamples: 1} made
     *  it a nested superset of the seeding sample instead of a second opinion about the data. */
    @Test
    void bernoulliStreamsCoverTheWholeInputAndAreIndependentOfEachOther() {
        int rows = 20_000;
        double probability = 0.05;
        DriverSample.BernoulliStreams streams =
            new DriverSample.BernoulliStreams(7L, 0, new double[] {probability, probability});

        Set<Integer> first = new HashSet<>();
        Set<Integer> second = new HashSet<>();
        for (int i = 0; i < rows; i++) {
            if (streams.take(0)) {
                first.add(i);
            }
            if (streams.take(1)) {
                second.add(i);
            }
        }

        int expected = (int) (rows * probability);
        for (Set<Integer> drawn : new Set[] {first, second}) {
            assertTrue(drawn.size() > expected * 0.8 && drawn.size() < expected * 1.2,
                "drew " + drawn.size() + ", expected about " + expected);

            double sum = 0.0;
            int max = 0;
            for (int index : drawn) {
                sum += index;
                max = Math.max(max, index);
            }
            assertTrue(sum / drawn.size() > 8_000.0 && sum / drawn.size() < 12_000.0,
                "draw mean " + (sum / drawn.size()) + " is not centred on the input");
            assertTrue(max > 18_000, "draw never reaches the tail of the input (max " + max + ")");
        }

        Set<Integer> shared = new HashSet<>(first);
        shared.retainAll(second);
        // Independent streams overlap in about p of each other; a shared generator would nest them.
        assertTrue(shared.size() < first.size() * 0.25,
            "streams overlap in " + shared.size() + " of " + first.size()
                + " draws — they are not independent");
    }
}
