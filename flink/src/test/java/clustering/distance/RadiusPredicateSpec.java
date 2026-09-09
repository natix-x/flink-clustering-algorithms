package clustering.distance;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** {@code withinRadius} is an optimisation of {@code compute(a, b) <= radius}, and
 *  {@code distanceUpTo} the same for a bounded nearest-prototype scan; the ε-scans of
 *  {@code dbscanpp} and the nearest-centroid loops of {@code kmeans} use nothing else — so a
 *  disagreement here is a wrong core point or a wrong assignment, i.e. a wrong cluster count,
 *  with no other symptom. This suite pins both to {@code compute} for every metric, mirroring
 *  the Spark repo's {@code RadiusPredicateSpec} so both engines are held to the same property. */
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

    // ── distanceUpTo ──────────────────────────────────────────────────────────────────────────
    // The nearest-prototype sibling of withinRadius, used by kmeans' nearest/second-nearest scan.
    // Same contract, one more thing to get right: when it returns a finite value that value must
    // be the true distance, because the caller keeps it as the running minimum (or second).

    @Test
    void distanceUpToReturnsTheTrueDistanceBelowTheBoundAndInfinityAboveIt() {
        Random random = new Random(23L);
        for (Map.Entry<String, DistanceMetric> entry : metrics().entrySet()) {
            for (int trial = 0; trial < 400; trial++) {
                int dim = 1 + random.nextInt(8);
                double[] a = new double[dim];
                double[] b = new double[dim];
                for (int i = 0; i < dim; i++) {
                    a[i] = random.nextDouble() * 4 - 2;
                    b[i] = random.nextDouble() * 4 - 2;
                }
                double bound = random.nextDouble() * 4;
                double exact = entry.getValue().compute(a, b);
                double bounded = entry.getValue().distanceUpTo(a, b, bound);
                if (exact <= bound) {
                    assertEquals(exact, bounded, 1e-9,
                        entry.getKey() + ": distanceUpTo returned " + bounded + ", compute says " + exact);
                } else {
                    assertEquals(Double.POSITIVE_INFINITY, bounded,
                        entry.getKey() + ": distanceUpTo returned " + bounded + " for a pair beyond bound " + bound);
                }
            }
        }
    }

    @Test
    void aPointExactlyOnTheBoundIsAHitAndReportsItsDistance() {
        double[] a = {0.0, 0.0};
        double[] b = {3.0, 4.0};   // euclidean 5, manhattan 7
        assertEquals(5.0, EuclideanDistance.INSTANCE.distanceUpTo(a, b, 5.0));
        assertEquals(Double.POSITIVE_INFINITY, EuclideanDistance.INSTANCE.distanceUpTo(a, b, 4.999999));
        assertEquals(7.0, ManhattanDistance.INSTANCE.distanceUpTo(a, b, 7.0));
        assertEquals(Double.POSITIVE_INFINITY, ManhattanDistance.INSTANCE.distanceUpTo(a, b, 6.999999));
    }

    @Test
    void distanceUpToEarlyExitDoesNotReportAHitBeforeTheLastCoordinateIsSeen() {
        int dim = 64;
        double[] a = new double[dim];
        double[] b = new double[dim];
        b[dim - 1] = 10.0;
        assertEquals(Double.POSITIVE_INFINITY, EuclideanDistance.INSTANCE.distanceUpTo(a, b, 9.0));
        assertEquals(10.0, EuclideanDistance.INSTANCE.distanceUpTo(a, b, 10.0));
        assertEquals(Double.POSITIVE_INFINITY, ManhattanDistance.INSTANCE.distanceUpTo(a, b, 9.0));
        assertEquals(10.0, ManhattanDistance.INSTANCE.distanceUpTo(a, b, 10.0));
    }

    /** NaN follows {@code compute}, not intuition. For the L-norms a NaN coordinate poisons the
     *  sum, every comparison against the bound is false and the pair is reported as beyond it.
     *  Cosine is the exception and legitimately so: {@code b} here is the zero vector, for which
     *  {@code compute} short-circuits to 1.0 without ever looking at {@code a}'s coordinates — so
     *  a finite answer is the correct one, and asserting infinity for every metric would be
     *  asserting a bug. */
    @Test
    void nanCoordinatesFollowComputeWhateverItSays() {
        double[] a = {0.0, Double.NaN};
        double[] b = {0.0, 0.0};
        for (Map.Entry<String, DistanceMetric> entry : metrics().entrySet()) {
            double exact = entry.getValue().compute(a, b);
            double bounded = entry.getValue().distanceUpTo(a, b, 1e9);
            if (exact <= 1e9) {
                assertEquals(exact, bounded, 1e-9, entry.getKey() + ": got " + bounded + ", compute says " + exact);
            } else {
                assertEquals(Double.POSITIVE_INFINITY, bounded, entry.getKey() + ": NaN slipped through as " + bounded);
            }
        }
    }

    /** An infinite bound must behave like an unbounded scan — that is the {@code assign: closest}
     *  path, where no ε caps the search. */
    @Test
    void anInfiniteBoundAlwaysReportsTheExactDistance() {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {-1.0, 0.5, 7.0};
        for (Map.Entry<String, DistanceMetric> entry : metrics().entrySet()) {
            double exact = entry.getValue().compute(a, b);
            assertEquals(exact, entry.getValue().distanceUpTo(a, b, Double.POSITIVE_INFINITY), 1e-9,
                entry.getKey() + ": an unbounded call did not match compute");
        }
    }
}
