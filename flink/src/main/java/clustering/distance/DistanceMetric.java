package clustering.distance;

import java.io.Serializable;

/** A distance between two feature vectors. Java mirror of the Spark repo's
 *  {@code clustering.distance.DistanceMetric}. */
public interface DistanceMetric extends Serializable {
    double compute(double[] a, double[] b);
}