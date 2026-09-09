package clustering.algorithms.kmedoids;

import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Model;
import clustering.distance.DistanceMetric;

/**
 * Fitted k-medoids model.
 * Cluster centers are actual data points (medoids) rather than computed centroids.
 */
public class KMedoidsModel implements Model {

    private final DenseVector[] medoids;
    private final DistanceMetric distanceMetric;

    public KMedoidsModel(DenseVector[] medoids, DistanceMetric distanceMetric) {
        this.medoids = medoids;
        this.distanceMetric = distanceMetric;
    }

    public DenseVector[] medoids() {
        return medoids;
    }

    @Override
    public int predict(DenseVector features) {
        int closestMedoidIndex = 0;
        double minDistance = Double.MAX_VALUE;

        for (int i = 0; i < medoids.length; i++) {
            double dist = distanceMetric.compute(features, medoids[i]);
            if (dist < minDistance) {
                minDistance = dist;
                closestMedoidIndex = i;
            }
        }
        return closestMedoidIndex;
    }
}
