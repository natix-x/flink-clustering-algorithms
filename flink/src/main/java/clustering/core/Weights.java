package clustering.core;

import java.util.Arrays;

/**
 * Utilities for handling point weights in clustering algorithms.
 */
public final class Weights {

    private Weights() {}

    /**
     * Generates an array of unit weights (1.0) for a given number of points.
     */
    public static double[] generateUnitWeights(int pointCount) {
        double[] unitWeights = new double[pointCount];
        Arrays.fill(unitWeights, 1.0);
        return unitWeights;
    }

    /**
     * Sanitizes a weight for safe aggregation.
     * Replaces NaN, infinite, and negative values with 0.0 to prevent calculation corruption.
     */
    public static double sanitize(double weight) {
        return (Double.isFinite(weight) && weight > 0.0) ? weight : 0.0;
    }
}
