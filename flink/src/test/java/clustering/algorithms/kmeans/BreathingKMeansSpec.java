package clustering.algorithms.kmeans;

import clustering.TestFixtures;
import clustering.core.Points;
import clustering.core.EnvFactory;
import clustering.core.EuclideanGeometry;
import clustering.core.PointSource;
import clustering.distance.EuclideanDistance;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The {@code refine: breathing} knob. Counterpart of the Spark {@code BreathingKMeansSpec}:
 *  breathing is only worth its extra passes if it escapes an initialisation plain Lloyd is
 *  stuck in, so that is what is asserted — not merely that it runs. */
class BreathingKMeansSpec {

    /** Three well-separated 4x4 blobs. With {@code seed = 42} the seeded init drops two
     *  centroids into one blob, so plain Lloyd converges with one centroid covering two blobs —
     *  a genuine local minimum, and the case breathing exists for. */
    private static List<double[]> threeBlobs() {
        List<double[]> points = new ArrayList<>();
        points.addAll(TestFixtures.grid(0.0, 0.0, 4, 0.2));
        points.addAll(TestFixtures.grid(20.0, 0.0, 4, 0.2));
        points.addAll(TestFixtures.grid(0.0, 20.0, 4, 0.2));
        return points;
    }

    private static int blobOf(double[] p) {
        if (p[0] > 10.0) {
            return 1;
        }
        return p[1] > 10.0 ? 2 : 0;
    }

    @Test
    void escapesALocalMinimumPlainLloydIsStuckIn() {
        List<double[]> points = threeBlobs();
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        KMeansModel plain = new KMeans(3, 20, 1e-4, 42L).fit(source, envs, 2);
        double plainError = TestFixtures.totalError(plain, points, plain.centroids(), EuclideanDistance.INSTANCE);
        assertTrue(TestFixtures.clustersOf(plain.labels(Points.vectorsOf(points))).size() == 3);
        // Sanity: the fixture really is a trap for this seed — some cluster spans two blobs.
        int[] plainLabels = plain.labels(Points.vectorsOf(points));
        boolean plainMergedTwoBlobs = false;
        for (int i = 0; i < points.size(); i++) {
            for (int j = 0; j < points.size(); j++) {
                if (plainLabels[i] == plainLabels[j] && blobOf(points.get(i)) != blobOf(points.get(j))) {
                    plainMergedTwoBlobs = true;
                }
            }
        }
        assertTrue(plainMergedTwoBlobs, "fixture broken: plain Lloyd already separates the blobs");

        KMeansModel breathing = new BreathingKMeans(3, 5, 20, 1e-4, 42L,
            EuclideanGeometry.INSTANCE, 10).fit(source, envs, 2);
        int[] labels = breathing.labels(Points.vectorsOf(points));
        double breathingError =
            TestFixtures.totalError(breathing, points, breathing.centroids(), EuclideanDistance.INSTANCE);

        assertEquals(3, breathing.centroids().length, "breathing must shrink back to k");
        assertTrue(breathingError < plainError,
            "breathing SSE " + breathingError + " should beat plain Lloyd's " + plainError);
        for (int i = 0; i < points.size(); i++) {
            for (int j = 0; j < points.size(); j++) {
                boolean sameBlob = blobOf(points.get(i)) == blobOf(points.get(j));
                assertEquals(sameBlob, labels[i] == labels[j],
                    "points " + i + "," + j + " grouped against their blobs");
            }
        }
    }

    /** Already-optimal input: breathing must not make things worse, and must still return k. */
    @Test
    void neverReturnsWorseThanItsStartingSolution() {
        List<double[]> points = threeBlobs();
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        KMeansModel plain = new KMeans(3, 20, 1e-4, 7L).fit(source, envs, 2);
        KMeansModel breathing = new BreathingKMeans(3, 2, 20, 1e-4, 7L,
            EuclideanGeometry.INSTANCE, 5).fit(source, envs, 2);

        double plainError = TestFixtures.totalError(plain, points, plain.centroids(), EuclideanDistance.INSTANCE);
        double breathingError =
            TestFixtures.totalError(breathing, points, breathing.centroids(), EuclideanDistance.INSTANCE);

        assertEquals(3, breathing.centroids().length);
        assertTrue(breathingError <= plainError + 1e-9,
            "breathing kept a worse solution: " + breathingError + " > " + plainError);
    }
}
