package clustering.algorithms.kmedoids;

import clustering.core.Model;
import clustering.distance.DistanceMetric;

/** Fitted medoid model. Java mirror of the Spark {@code KMedoidsModel}: cluster
 *  centres are actual data points (medoids) instead of arbitrary centroids. */
public class KMedoidsModel implements Model {

    private final double[][] medoids;
    private final DistanceMetric distance;

    public KMedoidsModel(double[][] medoids, DistanceMetric distance) {
        this.medoids = medoids;
        this.distance = distance;
    }

    public double[][] medoids() {
        return medoids;
    }

    @Override
    public int predict(double[] features) {
        int best = 0;
        double min = Double.MAX_VALUE;
        for (int j = 0; j < medoids.length; j++) {
            double d = distance.compute(features, medoids[j]);
            if (d < min) {
                min = d;
                best = j;
            }
        }
        return best;
    }
}