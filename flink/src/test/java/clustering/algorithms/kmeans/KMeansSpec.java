package clustering.algorithms.kmeans;

import clustering.TestFixtures;
import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Points;
import clustering.core.EnvFactory;
import clustering.core.EuclideanGeometry;
import clustering.core.PointSource;
import clustering.core.SphericalGeometry;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** K-means on the FLIP-176 iteration: cluster recovery, the {@code geometry} knob, and
 *  reproducibility. Counterpart of the Spark {@code LloydKMeansSpec} / {@code GeometrySpec}. */
class KMeansSpec {

    /** Three well-separated 4x4 blobs — deterministic by construction, so nothing can flake. */
    private static List<double[]> threeBlobs() {
        List<double[]> points = new ArrayList<>();
        points.addAll(TestFixtures.grid(0.0, 0.0, 4, 0.2));
        points.addAll(TestFixtures.grid(20.0, 0.0, 4, 0.2));
        points.addAll(TestFixtures.grid(0.0, 20.0, 4, 0.2));
        return points;
    }

    /** Blob index of a point, by construction of {@link #threeBlobs()}. */
    private static int blobOf(double[] p) {
        if (p[0] > 10.0) {
            return 1;
        }
        return p[1] > 10.0 ? 2 : 0;
    }

    @Test
    void recoversWellSeparatedBlobs() {
        List<double[]> points = threeBlobs();
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        KMeansModel model = new KMeans(3, 20, 1e-4, 7L).fit(source, envs, 2);
        int[] labels = model.labels(Points.vectorsOf(points));

        assertEquals(3, model.centroids().length);
        assertEquals(3, TestFixtures.clustersOf(labels).size(), "one cluster per blob");
        // Same blob -> same label, different blob -> different label.
        for (int i = 0; i < points.size(); i++) {
            for (int j = 0; j < points.size(); j++) {
                boolean sameBlob = blobOf(points.get(i)) == blobOf(points.get(j));
                assertEquals(sameBlob, labels[i] == labels[j],
                    "points " + i + "," + j + " grouped against their blobs");
            }
        }
    }

    @Test
    void isReproducibleAcrossRepeatedFits() {
        List<double[]> points = threeBlobs();
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        int[] first = new KMeans(3, 20, 1e-4, 7L).fit(source, envs, 2).labels(Points.vectorsOf(points));
        int[] second = new KMeans(3, 20, 1e-4, 7L).fit(source, envs, 2).labels(Points.vectorsOf(points));
        assertArrayEqualsLabels(first, second);
    }

    /** Spherical geometry clusters by DIRECTION: the same two rays at wildly different
     *  magnitudes must come out as two clusters, which Euclidean k-means would split by
     *  magnitude instead. */
    @Test
    void sphericalGeometryClustersByDirection() {
        List<double[]> points = Arrays.asList(
            new double[] {1.0, 0.0}, new double[] {50.0, 0.5}, new double[] {10.0, 0.1},
            new double[] {0.0, 1.0}, new double[] {0.5, 50.0}, new double[] {0.1, 10.0});
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        KMeansModel model = new KMeans(2, 20, 1e-4, 42L, SphericalGeometry.INSTANCE)
            .fit(source, envs, 2);
        int[] labels = model.labels(Points.vectorsOf(points));

        assertEquals(labels[0], labels[1], "x-ray points share a cluster regardless of magnitude");
        assertEquals(labels[0], labels[2]);
        assertEquals(labels[3], labels[4], "y-ray points share a cluster regardless of magnitude");
        assertEquals(labels[3], labels[5]);
        assertTrue(labels[0] != labels[3], "the two directions must not merge");
        for (DenseVector centroid : model.centroids()) {
            double[] c = centroid.values;
            double norm = Math.sqrt(c[0] * c[0] + c[1] * c[1]);
            assertEquals(1.0, norm, 1e-9, "spherical centroids stay on the unit sphere");
        }
    }

    /** k = 1 degenerates to the global mean, and the euclidean geometry is a no-op wrapper. */
    @Test
    void singleClusterIsTheGlobalMean() {
        List<double[]> points = threeBlobs();
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        KMeansModel model = new KMeans(1, 20, 1e-4, 7L, EuclideanGeometry.INSTANCE)
            .fit(source, envs, 2);

        double[] expected = new double[2];
        for (double[] p : points) {
            expected[0] += p[0] / points.size();
            expected[1] += p[1] / points.size();
        }
        assertEquals(expected[0], model.centroids()[0].values[0], 1e-6);
        assertEquals(expected[1], model.centroids()[0].values[1], 1e-6);
    }

    private static void assertArrayEqualsLabels(int[] a, int[] b) {
        assertEquals(a.length, b.length);
        for (int i = 0; i < a.length; i++) {
            assertEquals(a[i], b[i], "label " + i + " must be reproducible");
        }
    }
}
