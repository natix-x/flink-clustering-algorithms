package clustering.evaluation;

import clustering.TestFixtures;
import clustering.core.EnvFactory;
import clustering.core.PointSource;
import clustering.distance.CosineDistance;
import clustering.distance.DistanceMetric;
import clustering.distance.EuclideanDistance;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** The silhouette, pinned against values computed BY HAND from Rousseeuw's definition.
 *
 *  <p>It is the metric the thesis quotes most and the only O(n²) one, so both its arithmetic and its
 *  degenerate cases are fixed here rather than left to whatever the implementation happens to
 *  return. Every case and every expected number is the Spark {@code SilhouetteSpec}'s, which is the
 *  point: a silent divergence here would read as a genuine quality difference between the engines.
 *  The labelling is supplied by the test ({@link LabelByCoordinateModel}), so what is under test is
 *  the evaluator alone. */
class SilhouetteSpec {

    private static final int PARALLELISM = 2;
    private static final double TOLERANCE = 1e-9;

    private static final EnvFactory ENVS = TestFixtures.localEnvs(PARALLELISM);

    private static double silhouette(double[][] points, int[] labels) {
        return silhouette(points, labels, EuclideanDistance.INSTANCE);
    }

    private static double silhouette(double[][] points, int[] labels, DistanceMetric metric) {
        return new SilhouetteEvaluator(metric)
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

    /** {0, 2} and {10, 12}. a = 2 for every point; b = 11, 9, 9, 11 -> s = 9/11, 7/9, 7/9, 9/11,
     *  mean 79/99. */
    @Test
    void meanSilhouetteMatchesTheHandComputedValue() {
        assertEquals(79.0 / 99.0,
            silhouette(new double[][] {{0.0}, {2.0}, {10.0}, {12.0}}, new int[] {0, 0, 1, 1}),
            TOLERANCE);
    }

    /** The case that would otherwise need a synthetic row id: two DISTINCT rows sharing coordinates.
     *
     *  <p>Cluster {0, 0, 2}, plus {10}. For either point at 0, {@code a} must average over the OTHER
     *  two members — the twin at distance 0 and the point at distance 2 — giving 1, not the 2 that
     *  dropping every coordinate-equal row would give. Hand: s = 0.9, 0.9, 0.75, 0 -> 0.6375. */
    @Test
    void aCoordinateDuplicateNeighbourCountsOnlyThePointItselfIsExcluded() {
        assertEquals(0.6375,
            silhouette(new double[][] {{0.0}, {0.0}, {2.0}, {10.0}}, new int[] {0, 0, 0, 1}),
            TOLERANCE);
    }

    /** Rousseeuw: a point alone in its cluster scores 0, NOT 1. Treating its undefined {@code a} as
     *  0 would reward exactly the labellings (stray singletons out of a density or medoid run) the
     *  score is supposed to punish. Hand: 0.98, 96/98, 0. */
    @Test
    void aSingletonClusterContributesZeroNotOne() {
        assertEquals((0.98 + 96.0 / 98.0) / 3.0,
            silhouette(new double[][] {{0.0}, {2.0}, {100.0}}, new int[] {0, 0, 1}),
            TOLERANCE);
    }

    /** One cluster: {@code b} does not exist, so the score does not either. 0.0, never the ~1.0 that
     *  standing in an infinite {@code b} produces. */
    @Test
    void aSingleClusterYieldsZero() {
        assertEquals(0.0,
            silhouette(new double[][] {{0.0}, {2.0}, {9.0}}, new int[] {0, 0, 0}));
    }

    @Test
    void noisePointsAreExcluded() {
        assertEquals(79.0 / 99.0,
            silhouette(new double[][] {{0.0}, {2.0}, {10.0}, {12.0}, {500.0}, {-500.0}},
                new int[] {0, 0, 1, 1, -1, -1}),
            TOLERANCE);
    }

    /** Weighting == duplication, end to end: neighbour masses inside {@code a} and {@code b}, the
     *  cluster mass that divides them, and the weighted mean over rows. */
    @Test
    void weightingEqualsDuplication() {
        double[][] points = {{0.0}, {2.0}, {10.0}, {12.0}};
        int[] labels = {0, 0, 1, 1};
        double[] weights = {2.0, 1.0, 1.0, 2.0};

        List<double[]> duplicatedRows = new ArrayList<>();
        for (int i = 0; i < points.length; i++) {
            for (int copy = 0; copy < (int) weights[i]; copy++) {
                duplicatedRows.add(points[i].clone());
            }
        }
        LabelByCoordinateModel model = new LabelByCoordinateModel(points, labels);
        SilhouetteEvaluator evaluator = new SilhouetteEvaluator(EuclideanDistance.INSTANCE);

        assertEquals(
            evaluator.evaluate(model, TestFixtures.source(duplicatedRows), ENVS),
            evaluator.evaluate(model, TestFixtures.weightedSource(rows(points), weights), ENVS),
            TOLERANCE);
    }

    /** {@code d(p,p)} is subtracted, not assumed to be 0: under cosine a zero vector is at distance
     *  1.0 from everything INCLUDING itself. Two zero vectors in one cluster, two unit axes in the
     *  other: every a and every b is 1, so the score is exactly 0. Assuming a zero self-distance
     *  would inflate {@code a} to 2 and drag the score to -0.25. */
    @Test
    void cosineAZeroVectorsNonZeroSelfDistanceIsRemovedFromA() {
        assertEquals(0.0,
            silhouette(new double[][] {{0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}},
                new int[] {0, 0, 1, 1}, CosineDistance.INSTANCE),
            TOLERANCE);
    }
}
