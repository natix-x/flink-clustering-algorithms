package clustering.algorithms.dbscan;

import clustering.TestFixtures;
import clustering.core.EnvFactory;
import clustering.core.PointSource;
import clustering.distance.EuclideanDistance;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/** Step 2 of DBSCAN++: exact ε-degrees of every candidate against the FULL dataset.
 *
 *  The counts are checked against a brute-force O(n·m) loop, and — the reason this suite exists —
 *  against Flink's collect-sink record limit. The counting job used to hand its whole m-wide count
 *  array back as ONE record, which silently worked for every small m and then failed on a real run
 *  at m = 2·10⁶ ("Record size is 16000016 bytes, but max bytes per batch is only 2097152") AFTER
 *  half an hour of correct counting. The counts now leave the job one finished chunk at a time, so
 *  a record's size is tied to {@code chunkSize} and not to m. */
class EpsilonNeighbourCounterSpec {

    private static final double Eps = 2.0;

    private static List<double[]> points(int count, long seed) {
        Random random = new Random(seed);
        List<double[]> out = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            out.add(new double[] {random.nextDouble() * 20, random.nextDouble() * 20});
        }
        return out;
    }

    /** ε-degrees straight from the definition. */
    private static double[] bruteForce(List<double[]> data, double[][] candidates, double eps) {
        double[] counts = new double[candidates.length];
        for (int i = 0; i < candidates.length; i++) {
            for (double[] x : data) {
                if (EuclideanDistance.INSTANCE.compute(candidates[i], x) <= eps) {
                    counts[i] += 1.0;
                }
            }
        }
        return counts;
    }

    @Test
    void countsMatchTheDefinition() {
        List<double[]> data = points(1500, 4L);
        double[][] candidates = points(300, 9L).toArray(new double[0][]);
        PointSource source = TestFixtures.source(data);

        double[] counts = EpsilonNeighbourCounter.computeNeighbourhoodDensities(
            source, TestFixtures.localEnvs(2), candidates, Eps, EuclideanDistance.INSTANCE, 128);

        assertArrayEquals(bruteForce(data, candidates, Eps), counts, 1e-9);
    }

    /** The regression, reproduced cheaply: 300 000 candidates are 2.4 MB of counts, over Flink's
     *  2 MB collect-sink record limit — the exact failure a real {@code dbscanexact} run hit at
     *  m = 2·10⁶. The candidate count is what breaks it, NOT the data size, so 400 points are
     *  enough and the whole test is seconds rather than half an hour.
     *
     *  Chunking is what saves it: at 50 000 candidates per round each record is 400 KB. */
    @Test
    void aResultWiderThanTheCollectRecordLimitStillComesBack() {
        List<double[]> data = points(400, 4L);
        double[][] candidates = points(300_000, 9L).toArray(new double[0][]);
        PointSource source = TestFixtures.source(data);

        double[] counts = EpsilonNeighbourCounter.computeNeighbourhoodDensities(
            source, TestFixtures.localEnvs(2), candidates, Eps, EuclideanDistance.INSTANCE, 50_000);

        assertArrayEquals(bruteForce(data, candidates, Eps), counts, 1e-9);
    }

    /** Chunking is a memory bound, not an approximation: every chunk size must give the same
     *  counts, including the degenerate "one candidate per pass" and "everything in one pass". */
    @Test
    void chunkSizeDoesNotChangeTheCounts() {
        List<double[]> data = points(800, 4L);
        double[][] candidates = points(60, 9L).toArray(new double[0][]);
        PointSource source = TestFixtures.source(data);
        EnvFactory envs = TestFixtures.localEnvs(2);
        double[] expected = bruteForce(data, candidates, Eps);

        for (int chunkSize : new int[] {1, 7, 60, 5000}) {
            assertArrayEquals(expected, EpsilonNeighbourCounter.computeNeighbourhoodDensities(
                source, envs, candidates, Eps, EuclideanDistance.INSTANCE, chunkSize), 1e-9,
                "counts changed at chunkSize=" + chunkSize);
        }
    }

    /** A candidate that is itself a dataset point counts ITSELF — the DBSCAN core condition. */
    @Test
    void aCandidateCountsItself() {
        List<double[]> data = List.of(new double[] {0.0, 0.0}, new double[] {100.0, 100.0});
        double[][] candidates = {{0.0, 0.0}};

        double[] counts = EpsilonNeighbourCounter.computeNeighbourhoodDensities(
            TestFixtures.source(data), TestFixtures.localEnvs(2), candidates, Eps,
            EuclideanDistance.INSTANCE, 16);

        assertArrayEquals(new double[] {1.0}, counts, 1e-9);
    }
}
