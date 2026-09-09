package clustering.algorithms.kmedoids;

import clustering.algorithms.kmedoids.local.FastPAM;
import clustering.algorithms.kmedoids.local.FasterPAM;
import clustering.core.Weights;
import clustering.distance.EuclideanDistance;
import org.apache.flink.ml.linalg.DenseVector;
import org.junit.jupiter.api.Test;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Driver-local checks on the k-medoids ladder. No Flink cluster — every driver-local entry has a
 *  {@code fitLocal} path, which is exactly the path CLARA and the sampled runs use.
 *
 *  The point of the ladder is that FastPAM and FasterPAM optimise the SAME objective with
 *  progressively cheaper search, so they must land on comparable solutions; a variant that quietly
 *  optimises something else would show up here. Mirrors the Spark {@code KMedoidsLadderSpec},
 *  including the blob layout and the seed, so the two engines are checked against the same data. */
class KMedoidsLadderSpec {

    private static final double[][] BLOB_CENTRES = {{0.0, 0.0}, {50.0, 0.0}, {0.0, 50.0}};

    private final double[][] points;
    private final int[] blobOf;

    /** Three well-separated 2D blobs, 60 points each, fixed seed. */
    KMedoidsLadderSpec() {
        int perBlob = 60;
        points = new double[BLOB_CENTRES.length * perBlob][];
        blobOf = new int[points.length];
        Random rnd = new Random(7L);
        int at = 0;
        for (int blob = 0; blob < BLOB_CENTRES.length; blob++) {
            for (int i = 0; i < perBlob; i++) {
                points[at] = new double[] {
                    BLOB_CENTRES[blob][0] + rnd.nextGaussian(),
                    BLOB_CENTRES[blob][1] + rnd.nextGaussian()
                };
                blobOf[at] = blob;
                at++;
            }
        }
    }

    private double cost(DenseVector[] medoids) {
        double total = 0.0;
        for (double[] point : points) {
            double nearest = Double.MAX_VALUE;
            for (DenseVector medoid : medoids) {
                nearest = Math.min(nearest, EuclideanDistance.INSTANCE.compute(point, medoid.values));
            }
            total += nearest;
        }
        return total;
    }

    private Set<Integer> blobsOf(DenseVector[] medoids) {
        Set<Integer> blobs = new HashSet<>();
        for (DenseVector medoid : medoids) {
            int found = -1;
            for (int i = 0; i < points.length && found < 0; i++) {
                if (points[i] == medoid.values) {
                    found = i;
                }
            }
            assertTrue(found >= 0, "medoid must be one of the input points");
            blobs.add(blobOf[found]);
        }
        return blobs;
    }

    @Test
    void fastPamRecoversOneMedoidPerBlob() {
        DenseVector[] medoids =
            new FastPAM(3, 100, EuclideanDistance.INSTANCE).fitLocal(points).medoids();
        assertEquals(3, medoids.length);
        assertEquals(Set.of(0, 1, 2), blobsOf(medoids),
            "the exact rung must separate three well-separated blobs");
    }

    @Test
    void fasterPamRecoversOneMedoidPerBlob() {
        DenseVector[] medoids =
            new FasterPAM(3, 100, EuclideanDistance.INSTANCE, 42L).fitLocal(points).medoids();
        assertEquals(3, medoids.length);
        assertEquals(Set.of(0, 1, 2), blobsOf(medoids),
            "eager swapping must still separate three well-separated blobs");
    }

    @Test
    void fastPamAndFasterPamAgreeOnTheSameObjective() {
        DenseVector[] fastPam =
            new FastPAM(3, 100, EuclideanDistance.INSTANCE).fitLocal(points).medoids();
        DenseVector[] fasterPam =
            new FasterPAM(3, 100, EuclideanDistance.INSTANCE, 42L).fitLocal(points).medoids();

        // Same objective, two different searches: FastPAM takes the best swap per iteration,
        // FasterPAM every improving one. On separated blobs both must still reach the same local
        // optimum, so the costs coincide up to floating-point noise. Equal COST is the right
        // assertion for the eager variant — a different equally-good optimum is a legitimate
        // outcome for it, unlike for FastPAM, whose search is PAM's exactly.
        assertTrue(Math.abs(cost(fasterPam) - cost(fastPam)) < 1e-9,
            "FasterPAM cost " + cost(fasterPam) + " != FastPAM cost " + cost(fastPam));
    }

    @Test
    void fasterPamIsDeterministicForAFixedSeed() {
        assertEquals(medoidCoordinates(42L), medoidCoordinates(42L),
            "same seed must give the same medoids in the same order");
    }

    private String medoidCoordinates(long seed) {
        StringBuilder out = new StringBuilder();
        for (DenseVector medoid : new FasterPAM(3, 100, EuclideanDistance.INSTANCE, seed)
                .fitLocal(points).medoids()) {
            out.append(medoid).append(';');
        }
        return out.toString();
    }

    /** The weighted path must reduce to the unweighted one at unit weights — the invariant that
     *  lets every Σ-over-points in the ladder carry weights without an "is this weighted" branch. */
    @Test
    void unitWeightsReproduceTheUnweightedFit() {
        double[] unit = clustering.core.Weights.unit(points.length);
        assertEquals(
            cost(new FastPAM(3, 100, EuclideanDistance.INSTANCE).fitLocal(points).medoids()),
            cost(new FastPAM(3, 100, EuclideanDistance.INSTANCE).fitLocal(points, unit).medoids()));
    }
}
