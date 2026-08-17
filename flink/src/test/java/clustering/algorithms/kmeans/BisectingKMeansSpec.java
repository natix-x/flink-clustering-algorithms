package clustering.algorithms.kmeans;

import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Points;
import clustering.TestFixtures;
import clustering.core.EnvFactory;
import clustering.core.PointSource;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Bisecting k-means: the divisive split search as one FLIP-176 job. */
class BisectingKMeansSpec {

    /** Four well-separated 3x3 blobs, one per quadrant of a large square. */
    private static List<double[]> fourBlobs() {
        List<double[]> points = new ArrayList<>();
        points.addAll(TestFixtures.grid(0.0, 0.0, 3, 0.2));
        points.addAll(TestFixtures.grid(30.0, 0.0, 3, 0.2));
        points.addAll(TestFixtures.grid(0.0, 30.0, 3, 0.2));
        points.addAll(TestFixtures.grid(30.0, 30.0, 3, 0.2));
        return points;
    }

    private static int blobOf(double[] p) {
        return (p[0] > 15.0 ? 1 : 0) + (p[1] > 15.0 ? 2 : 0);
    }

    @Test
    void splitsIntoOneLeafPerBlob() {
        List<double[]> points = fourBlobs();
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        BisectingKMeansModel model = new BisectingKMeans(4, 20, 1e-4, 42L).fit(source, envs, 2);
        int[] labels = model.labels(Points.wrapAll(points));

        assertEquals(4, model.numClusters());
        assertEquals(4, TestFixtures.clustersOf(labels).size(), "one leaf per blob");
        for (int i = 0; i < points.size(); i++) {
            for (int j = 0; j < points.size(); j++) {
                boolean sameBlob = blobOf(points.get(i)) == blobOf(points.get(j));
                assertEquals(sameBlob, labels[i] == labels[j],
                    "points " + i + "," + j + " grouped against their blobs");
            }
        }
        // Leaf ids are DFS-numbered, so they are contiguous from 0.
        for (int label : labels) {
            assertTrue(label >= 0 && label < 4, "leaf ids must be contiguous from 0, got " + label);
        }
    }

    @Test
    void kEqualsOneIsTheGlobalMean() {
        List<double[]> points = fourBlobs();
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        BisectingKMeansModel model = new BisectingKMeans(1, 20, 1e-4, 42L).fit(source, envs, 2);
        assertEquals(1, model.numClusters());
        double[] expected = new double[2];
        for (double[] p : points) {
            expected[0] += p[0] / points.size();
            expected[1] += p[1] / points.size();
        }
        assertEquals(expected[0], model.clusterCentroids()[0].values[0], 1e-6);
        assertEquals(expected[1], model.clusterCentroids()[0].values[1], 1e-6);
    }

    /** Reproducibility is asserted on DIFFERENTLY-SIZED blobs on purpose. Flink's rebalance
     *  partitioner picks a random starting channel per run, so which subtask sees which point
     *  varies between runs; with four congruent blobs the leaf costs are exact ties up to
     *  floating-point summation order, and the split ORDER can then legitimately flip. Unequal
     *  blobs give the split criterion a real gap to resolve, which is what reproducibility means
     *  here (see the class docstring of {@link BisectingKMeans}). */
    @Test
    void isReproducibleAcrossRepeatedFits() {
        List<double[]> points = new ArrayList<>();
        points.addAll(TestFixtures.grid(0.0, 0.0, 2, 0.2));
        points.addAll(TestFixtures.grid(30.0, 0.0, 3, 0.4));
        points.addAll(TestFixtures.grid(0.0, 30.0, 4, 0.6));
        points.addAll(TestFixtures.grid(30.0, 30.0, 5, 0.8));
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        int[] first = new BisectingKMeans(4, 20, 1e-4, 42L).fit(source, envs, 2).labels(Points.wrapAll(points));
        int[] second = new BisectingKMeans(4, 20, 1e-4, 42L).fit(source, envs, 2).labels(Points.wrapAll(points));
        for (int i = 0; i < first.length; i++) {
            assertEquals(first[i], second[i], "label " + i + " must be reproducible");
        }
    }

    /** More clusters requested than distinct point groups the tree can produce: the search must
     *  stop early with a valid tree instead of looping or crashing. */
    @Test
    void stopsEarlyWhenNoLeafCanBeSplit() {
        List<double[]> points = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            points.add(new double[] {1.0, 1.0});
        }
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        BisectingKMeansModel model = new BisectingKMeans(3, 20, 1e-4, 42L).fit(source, envs, 2);
        assertEquals(1, model.numClusters(), "identical points cannot be bisected");
        for (int label : model.labels(Points.wrapAll(points))) {
            assertEquals(0, label);
        }
    }
}
