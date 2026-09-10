package clustering.core;

import clustering.distance.DistanceMetric;
import org.apache.flink.ml.linalg.DenseVector;

import java.io.Serializable;

/**
 * Defines the geometric space for centroid-based clustering algorithms.
 * Encapsulates data preparation, distance metrics, and centroid projections.
 */
public interface Geometry extends Serializable {

    String name();

    /**
     * Wraps the source with necessary spatial transformations (e.g., normalization).
     * Transformations are evaluated lazily as map operators.
     */
    PointSource prepare(PointSource source);

    DistanceMetric fitDistance();

    DistanceMetric modelDistance();

    DenseVector project(DenseVector centroid);

    /**
     * Returns a new L2-normalized array. Zero vectors are returned unchanged.
     */
    static double[] l2Normalize(double[] vector) {
        double sumOfSquares = 0.0;
        for (double value : vector) {
            sumOfSquares += value * value;
        }

        double norm = Math.sqrt(sumOfSquares);
        if (norm == 0.0) {
            return vector;
        }

        double[] normalizedVector = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalizedVector[i] = vector[i] / norm;
        }

        return normalizedVector;
    }
}
