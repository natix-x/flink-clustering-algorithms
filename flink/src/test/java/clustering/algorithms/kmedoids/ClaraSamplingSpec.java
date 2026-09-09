package clustering.algorithms.kmedoids;

import clustering.algorithms.kmedoids.components.DriverSample;
import clustering.algorithms.kmedoids.hybrid.CLARA;
import clustering.TestFixtures;
import clustering.core.EnvFactory;
import clustering.core.PointSource;
import clustering.distance.EuclideanDistance;
import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.WeightedPoint;
import clustering.core.WeightedPointTypeInfo;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** CLARA's samples must be draws over the WHOLE dataset, not a prefix of the stream. Mirrors the
 *  Spark {@code DistributedMedoidsSpec} sampling tests, which exist because the Spark side carried
 *  the same defect.
 *
 *  <p>The input is ORDERED — x grows monotonically with the row index — so position in the input IS
 *  the value, and the two ways the draw can break are both visible in the fitted medoid:
 *  <ul>
 *    <li>keeping the first {@code sampleSize} records off the collect iterator ({@code collectUpTo})
 *        yields a sample from the head of the data;</li>
 *    <li>concatenating per-subtask draws and truncating in subtask order drops every subtask after
 *        the first, so the sample covers one partition instead of the dataset.</li>
 *  </ul>
 *  {@code k = 1} makes the medoid of a sample its own centre point, so the sample's location is
 *  readable straight off the model — the only place sampling is observable through the public API. */
class ClaraSamplingSpec {

    private static final int ROWS = 20_000;
    private static final int PARALLELISM = 4;

    /** 20 000 rows, x strictly increasing, split into CONTIGUOUS ranges per subtask.
     *
     *  Contiguity is the whole point and {@code TestFixtures.source} cannot provide it: a
     *  {@code fromCollection} source has parallelism 1, so its records reach a parallel downstream
     *  round-robin — subtask 0 gets rows 0, 4, 8, … and therefore already spans the range, which
     *  would make every partition-order bias invisible. {@code fromSequence} splits the range into
     *  contiguous blocks, one per subtask, exactly as a parquet reader splits a file — the shape
     *  the real datasets have and the one a positional bias actually bites on. */
    private static PointSource orderedSource(int parallelism) {
        return env -> env.fromSequence(0, ROWS - 1)
            .setParallelism(parallelism)
            .map(i -> WeightedPoint.of(new double[] {i, 0.0}))
            .returns(WeightedPointTypeInfo.INSTANCE)
            .setParallelism(parallelism);
    }

    /** Every sample spans the whole range, so a k=1 medoid must land near the middle (10 000) —
     *  not in the first fifth, as a prefix-biased draw would. */
    private void assertCentred(DenseVector medoid, String what) {
        double x = medoid.values[0];
        assertTrue(x > 6_000.0 && x < 14_000.0,
            what + " medoid " + x + " is not a draw from the whole range — sampling is positionally biased");
    }

    /** Both parallelisms are checked, because the two defects show up at different ones. At
     *  parallelism 1 a stream prefix is literally the head of the data. At parallelism 4 a prefix of
     *  the INTERLEAVED collect order draws from the head of each of the four partitions, whose
     *  midpoint sits near the dataset's — so the prefix bug can hide there, while the combiner's
     *  subtask-order truncation (which keeps subtask 0 alone, i.e. the first quarter) shows up
     *  precisely there and not at parallelism 1. */
    private static final int[] PARALLELISMS = {1, PARALLELISM};

    @Test
    void claraSamplesTheWholeDatasetNotAPrefix() {
        for (int parallelism : PARALLELISMS) {
            EnvFactory envs = TestFixtures.localEnvs(parallelism);
            for (long seed : new long[] {1L, 7L, 23L}) {
                KMedoidsModel model = (KMedoidsModel) new CLARA(
                    1, 6, 500, 50, EuclideanDistance.INSTANCE, "fastpam", seed)
                    .fit(orderedSource(parallelism), envs, parallelism);
                assertCentred(model.medoids()[0], "clara (p=" + parallelism + ", seed " + seed + ")");
            }
        }
    }

    /** Shuffle-then-truncate is the whole fix, so pin it directly: the kept rows must span the
     *  input, and truncating a prefix instead would keep only the head. */
    @Test
    void takeRandomKeepsASpreadSubsetNotAPrefix() {
        List<double[]> rows = new ArrayList<>();
        for (int i = 0; i < ROWS; i++) {
            rows.add(new double[] {i, 0.0});
        }
        double[][] kept = DriverSample.takeRandom(rows, 500, 5L);

        double max = 0.0;
        double sum = 0.0;
        for (double[] row : kept) {
            max = Math.max(max, row[0]);
            sum += row[0];
        }
        double mean = sum / kept.length;
        assertTrue(mean > 8_000.0 && mean < 12_000.0, "kept mean " + mean + " is not centred");
        assertTrue(max > 18_000.0, "kept rows never reach the tail of the input (max " + max + ")");
    }
}