package clustering.distance;

import org.apache.flink.ml.linalg.DenseVector;

import java.io.Serializable;

/**
 * Computes the distance between two feature vectors.
 */
public interface DistanceMetric extends Serializable {

    double compute(double[] vectorA, double[] vectorB);

    /**
     * Computes the distance between two DenseVectors without copying the underlying arrays.
     */
    default double compute(DenseVector vectorA, DenseVector vectorB) {
        return compute(vectorA.values, vectorB.values);
    }

    /**
     * Checks if the distance between two vectors is less than or equal to the given radius.
     * Implementations can override this to provide an optimized, early-exit evaluation.
     */
    default boolean withinRadius(double[] vectorA, double[] vectorB, double radius) {
        return compute(vectorA, vectorB) <= radius;
    }

    /**
     * Computes the distance if it is less than or equal to the given bound.
     * Returns Double.POSITIVE_INFINITY otherwise.
     * Implementations can override this to optimize nearest-neighbor searches via early exit.
     */
    default double distanceUpTo(double[] vectorA, double[] vectorB, double bound) {
        double distance = compute(vectorA, vectorB);
        return distance <= bound ? distance : Double.POSITIVE_INFINITY;
    }
}
