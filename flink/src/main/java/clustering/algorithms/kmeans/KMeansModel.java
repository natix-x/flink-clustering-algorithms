package clustering.algorithms.kmeans;

import clustering.core.Model;
import clustering.distance.DistanceMetric;

/** Fitted centroid model. Mirrors the Spark {@code KMeansModel}. */
public class KMeansModel implements Model {

    private final double[][] centroids;
    private final DistanceMetric distance;

    public KMeansModel(double[][] centroids, DistanceMetric distance) {
        this.centroids = centroids;
        this.distance = distance;
    }

    public double[][] centroids() {
        return centroids;
    }

    @Override
    public int predict(double[] features) {
        int best = 0;
        double min = Double.MAX_VALUE;
        for (int j = 0; j < centroids.length; j++) {
            double d = distance.compute(features, centroids[j]);
            if (d < min) {
                min = d;
                best = j;
            }
        }
        return best;
    }
}