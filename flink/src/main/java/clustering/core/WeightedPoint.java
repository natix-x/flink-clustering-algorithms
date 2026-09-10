package clustering.core;

import org.apache.flink.ml.linalg.DenseVector;

/**
 * Represents a data point and its weight in the clustering space.
 * A weight indicates how much the point contributes to distance/objective calculations
 * (e.g., a weight of 3 is equivalent to three identical points).
 *
 * Note: Public mutable fields and a no-arg constructor are required for Flink object reuse and serialization.
 */
public final class WeightedPoint {

    public DenseVector features;
    public double weight;

    public WeightedPoint() {}

    public WeightedPoint(DenseVector features, double weight) {
        this.features = features;
        this.weight = weight;
    }

    /**
     * Creates a point with a default unit weight (1.0).
     */
    public static WeightedPoint withUnitWeight(DenseVector features) {
        return new WeightedPoint(features, 1.0);
    }

    public static WeightedPoint withUnitWeight(double[] features) {
        return new WeightedPoint(new DenseVector(features), 1.0);
    }

    /**
     * Returns the underlying coordinate array without copying.
     */
    public double[] getCoordinates() {
        return features.values;
    }

    /**
     * Returns the dimensionality of the point.
     */
    public int getDimensions() {
        return features.size();
    }

    @Override
    public String toString() {
        return "WeightedPoint{features=" + features + ", weight=" + weight + '}';
    }
}
