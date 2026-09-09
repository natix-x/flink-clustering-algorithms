package clustering.evaluation;

import clustering.TestFixtures;
import clustering.core.EnvFactory;
import clustering.core.PointSource;
import clustering.distance.EuclideanDistance;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The two centroid-based internal indices, pinned against values computed BY HAND from the
 *  published definitions — the same role {@code DBSCANppSpec}'s textbook DBSCAN plays for the
 *  density slot. Both are reported per run and compared across engines, so a silent formula slip
 *  here would look like a genuine quality difference between Spark and Flink.
 *
 *  <p>Mirrors the Spark {@code InternalIndicesSpec} case for case, and deliberately asserts the
 *  SAME hand-computed numbers: that is what makes the two engines' {@code daviesBouldin} /
 *  {@code calinskiHarabasz} columns comparable rather than merely both plausible. The labelling is
 *  fixed by the test ({@link LabelByCoordinateModel}), so what is under test is the index
 *  arithmetic and the two-pass distributed moment computation, never a clusterer. */
class InternalIndicesSpec {

    private static final int PARALLELISM = 2;
    private static final double TOLERANCE = 1e-9;

    private static final EnvFactory ENVS = TestFixtures.localEnvs(PARALLELISM);

    /** Two clusters of two points on a line: {0, 2} and {10, 12}.
     *  Centroids 1 and 11, S_0 = S_1 = 1, M_01 = 10  ->  DB = (1+1)/10 = 0.2.
     *  Grand centroid 6: between = 2·5² + 2·5² = 100, within = 4·1² = 4,
     *  k = 2, n = 4  ->  CH = (100/1) / (4/2) = 50. */
    private static final double[][] TWO_PAIRS = {{0.0}, {2.0}, {10.0}, {12.0}};
    private static final int[] TWO_PAIRS_LABELS = {0, 0, 1, 1};

    private static double daviesBouldin(double[][] points, int[] labels) {
        return new DaviesBouldinEvaluator(EuclideanDistance.INSTANCE)
            .evaluate(new LabelByCoordinateModel(points, labels), source(points), ENVS);
    }

    private static double calinskiHarabasz(double[][] points, int[] labels) {
        return new CalinskiHarabaszEvaluator(EuclideanDistance.INSTANCE)
            .evaluate(new LabelByCoordinateModel(points, labels), source(points), ENVS);
    }

    private static PointSource source(double[][] points) {
        return TestFixtures.source(rows(points));
    }

    private static List<double[]> rows(double[][] points) {
        List<double[]> out = new ArrayList<>(points.length);
        for (double[] point : points) {
            out.add(point.clone());
        }
        return out;
    }

    @Test
    void daviesBouldinMatchesTheHandComputedValue() {
        assertEquals(0.2, daviesBouldin(TWO_PAIRS, TWO_PAIRS_LABELS), TOLERANCE);
    }

    @Test
    void calinskiHarabaszMatchesTheHandComputedValue() {
        assertEquals(50.0, calinskiHarabasz(TWO_PAIRS, TWO_PAIRS_LABELS), TOLERANCE);
    }

    /** Direction of each index — the property the analysis actually relies on, and the one a sign
     *  or an inverted ratio would break while the magnitudes still look plausible. */
    @Test
    void separatingTheClustersLowersDaviesBouldinAndRaisesCalinskiHarabasz() {
        double[][] overlapping = {{0.0}, {2.0}, {1.0}, {3.0}};
        int[] labels = {0, 0, 1, 1};

        assertTrue(daviesBouldin(TWO_PAIRS, TWO_PAIRS_LABELS) < daviesBouldin(overlapping, labels));
        assertTrue(calinskiHarabasz(TWO_PAIRS, TWO_PAIRS_LABELS) > calinskiHarabasz(overlapping, labels));
    }

    /** Weighting == duplication, the repo-wide invariant: both indices aggregate MASS, so a point
     *  of weight w must score exactly as w copies of it. */
    @Test
    void weightingEqualsDuplicationForBothIndices() {
        double[][] points = {{0.0}, {2.0}, {10.0}, {12.0}};
        int[] labels = {0, 0, 1, 1};
        double[] weights = {3.0, 1.0, 1.0, 2.0};

        List<double[]> duplicatedRows = new ArrayList<>();
        for (int i = 0; i < points.length; i++) {
            for (int copy = 0; copy < (int) weights[i]; copy++) {
                duplicatedRows.add(points[i].clone());
            }
        }
        LabelByCoordinateModel model = new LabelByCoordinateModel(points, labels);
        PointSource weighted = TestFixtures.weightedSource(rows(points), weights);
        PointSource duplicated = TestFixtures.source(duplicatedRows);

        assertEquals(
            new DaviesBouldinEvaluator(EuclideanDistance.INSTANCE).evaluate(model, duplicated, ENVS),
            new DaviesBouldinEvaluator(EuclideanDistance.INSTANCE).evaluate(model, weighted, ENVS),
            TOLERANCE);
        assertEquals(
            new CalinskiHarabaszEvaluator(EuclideanDistance.INSTANCE).evaluate(model, duplicated, ENVS),
            new CalinskiHarabaszEvaluator(EuclideanDistance.INSTANCE).evaluate(model, weighted, ENVS),
            TOLERANCE);
    }

    /** Noise (label -1) is excluded, exactly as it is from the silhouette — otherwise a DBSCAN run
     *  would be scored on a "cluster" made of everything the algorithm refused to cluster. */
    @Test
    void noisePointsAreExcludedFromBothIndices() {
        double[][] withNoise = {{0.0}, {2.0}, {10.0}, {12.0}, {500.0}, {-500.0}};
        int[] labels = {0, 0, 1, 1, -1, -1};

        assertEquals(0.2, daviesBouldin(withNoise, labels), TOLERANCE);
        assertEquals(50.0, calinskiHarabasz(withNoise, labels), TOLERANCE);
    }

    /** Fewer than two clusters: both indices are undefined and report 0.0, the convention the other
     *  evaluators already use for "could not be computed". */
    @Test
    void aSingleClusterYieldsZeroForBothIndices() {
        double[][] single = {{0.0}, {2.0}};
        int[] labels = {0, 0};

        assertEquals(0.0, daviesBouldin(single, labels));
        assertEquals(0.0, calinskiHarabasz(single, labels));
    }

    /** Coincident centroids: the pair contributes nothing instead of an Infinity that would not even
     *  serialise as JSON (scikit-learn's {@code davies_bouldin_score} does the same).
     *
     *  <p>Asserted on hand-built moments rather than through a source, because the labelling the
     *  Spark spec uses — the same two coordinates in two different clusters — is not expressible
     *  through a coordinate-keyed model. What it exercises is the index, which is where the guard
     *  lives. */
    @Test
    void daviesBouldinStaysFiniteWhenTwoClustersShareACentroid() {
        ClusterMoments coincident = new ClusterMoments(
            new ClusterMoments.ClusterMoment[] {
                new ClusterMoments.ClusterMoment(0, new double[] {1.0}, 2.0, 2.0, 2.0),
                new ClusterMoments.ClusterMoment(1, new double[] {1.0}, 2.0, 2.0, 2.0)
            },
            new double[] {1.0});

        assertEquals(0.0, DaviesBouldinIndex.of(coincident, EuclideanDistance.INSTANCE));
    }

    /** Zero within-cluster dispersion makes the variance ratio's denominator vanish; scikit-learn
     *  reports 1.0 and so does this, on both engines. */
    @Test
    void calinskiHarabaszReportsOneWhenEveryPointSitsOnItsCentroid() {
        ClusterMoments degenerate = new ClusterMoments(
            new ClusterMoments.ClusterMoment[] {
                new ClusterMoments.ClusterMoment(0, new double[] {0.0}, 3.0, 0.0, 0.0),
                new ClusterMoments.ClusterMoment(1, new double[] {10.0}, 3.0, 0.0, 0.0)
            },
            new double[] {5.0});

        assertEquals(1.0, CalinskiHarabaszIndex.of(degenerate, EuclideanDistance.INSTANCE));
    }

    /** The moments themselves, since both indices are only as right as their inputs: ascending by
     *  label, weighted centroids, Σ w·d and Σ w·d² against those centroids. */
    @Test
    void clusterMomentsAreAscendingByLabelAndWeighted() {
        double[][] points = {{0.0}, {2.0}, {10.0}, {12.0}};
        int[] labels = {1, 1, 0, 0};
        double[] weights = {3.0, 1.0, 1.0, 1.0};

        ClusterMoments moments = ClusterMoments.compute(
            new LabelByCoordinateModel(points, labels),
            TestFixtures.weightedSource(rows(points), weights),
            ENVS, EuclideanDistance.INSTANCE);

        assertEquals(2, moments.numClusters());
        assertEquals(Arrays.asList(0, 1),
            Arrays.asList(moments.clusterMoments[0].label, moments.clusterMoments[1].label));
        // Cluster 1 = {0.0 (w=3), 2.0 (w=1)}: mass 4, centroid 0.5, Σ w·d = 3·0.5 + 1·1.5 = 3.0.
        assertEquals(4.0, moments.clusterMoments[1].clusterWeight, TOLERANCE);
        assertEquals(0.5, moments.clusterMoments[1].centroid[0], TOLERANCE);
        assertEquals(3.0, moments.clusterMoments[1].sumOfWeightedDistances, TOLERANCE);
        assertEquals(3.0 * 0.25 + 1.0 * 2.25, moments.clusterMoments[1].sumOfWeightedSquaredDistances, TOLERANCE);
        // Grand centroid = (3·0 + 1·2 + 1·10 + 1·12) / 6 = 4.0.
        assertEquals(4.0, moments.globalCentroid[0], TOLERANCE);
        assertEquals(6.0, moments.totalWeight(), TOLERANCE);
    }
}
