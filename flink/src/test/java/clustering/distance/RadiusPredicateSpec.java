package clustering.distance;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** {@code withinRadius} is an optimisation of {@code compute(a, b) <= radius}, and the ε-scans of
 *  {@code dbscanpp} use nothing else — so a disagreement here is a wrong core point, i.e. a wrong
 *  cluster count, with no other symptom. This suite pins it to {@code compute} for every metric,
 *  mirroring the Spark repo's {@code RadiusPredicateSpec} so both engines are held to the same
 *  property. */
class RadiusPredicateSpec {

    private static Map<String, DistanceMetric> metrics() {
        Map<String, DistanceMetric> m = new LinkedHashMap<>();
        m.put("euclidean", EuclideanDistance.INSTANCE);
        m.put("manhattan", ManhattanDistance.INSTANCE);
        m.put("cosine", CosineDistance.INSTANCE);
        m.put("unit-sphere", UnitSphereDistance.INSTANCE);
        return m;
    }

    @Test
    void agreesWithComputeOverRandomPairsAndRadii() {
        Random random = new Random(17L);
        for (Map.Entry<String, DistanceMetric> entry : metrics().entrySet()) {
            for (int trial = 0; trial < 400; trial++) {
                int dim = 1 + random.nextInt(8);
                double[] a = new double[dim];
                double[] b = new double[dim];
                for (int i = 0; i < dim; i++) {
                    a[i] = random.nextDouble() * 4 - 2;
                    b[i] = random.nextDouble() * 4 - 2;
                }
                double radius = random.nextDouble() * 4;
                boolean expected = entry.getValue().compute(a, b) <= radius;
                assertEquals(expected, entry.getValue().withinRadius(a, b, radius),
                    entry.getKey() + ": disagreed at r=" + radius);
            }
        }
    }

    @Test
    void aPointExactlyOnTheRadiusIsInsideIt() {
        double[] a = {0.0, 0.0};
        double[] b = {3.0, 4.0};   // euclidean 5, manhattan 7
        assertTrue(EuclideanDistance.INSTANCE.withinRadius(a, b, 5.0));
        assertFalse(EuclideanDistance.INSTANCE.withinRadius(a, b, 4.999999));
        assertTrue(ManhattanDistance.INSTANCE.withinRadius(a, b, 7.0));
        assertFalse(ManhattanDistance.INSTANCE.withinRadius(a, b, 6.999999));
    }

    /** A pair that only exceeds the radius in its LAST coordinate: exiting early on a partial sum
     *  must not report "inside" before that coordinate is seen. */
    @Test
    void theEarlyExitDoesNotChangeTheVerdict() {
        int dim = 64;
        double[] a = new double[dim];
        double[] b = new double[dim];
        b[dim - 1] = 10.0;
        assertFalse(EuclideanDistance.INSTANCE.withinRadius(a, b, 9.0));
        assertTrue(EuclideanDistance.INSTANCE.withinRadius(a, b, 10.0));
        assertFalse(ManhattanDistance.INSTANCE.withinRadius(a, b, 9.0));
        assertTrue(ManhattanDistance.INSTANCE.withinRadius(a, b, 10.0));
    }

    @Test
    void nanCoordinatesAreOutsideEveryRadius() {
        double[] a = {0.0, Double.NaN};
        double[] b = {0.0, 0.0};
        for (Map.Entry<String, DistanceMetric> entry : metrics().entrySet()) {
            boolean expected = entry.getValue().compute(a, b) <= 1e9;
            assertEquals(expected, entry.getValue().withinRadius(a, b, 1e9),
                entry.getKey() + ": NaN handling diverged");
        }
    }
}
