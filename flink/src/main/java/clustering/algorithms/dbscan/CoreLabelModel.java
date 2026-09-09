package clustering.algorithms.dbscan;

import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Model;
import clustering.distance.DistanceMetric;
import java.util.Arrays;

/**
 * Fitted DBSCAN++ density model containing labeled core points.
 * Assigns new points to the nearest core point's cluster, optionally enforcing the epsilon bound.
 */
public class CoreLabelModel implements Model {

    private final double[][] corePoints;
    private final int[] clusterLabels;
    private final double epsilon;
    private final DistanceMetric distanceMetric;
    private final boolean strictEpsilonCheck;

    public CoreLabelModel(double[][] corePoints, int[] clusterLabels, double epsilon,
                          DistanceMetric distanceMetric, boolean strictEpsilonCheck) {
        if (corePoints.length != clusterLabels.length) {
            throw new IllegalArgumentException(
                String.format("Core points count (%d) must match cluster labels count (%d)",
                    corePoints.length, clusterLabels.length));
        }
        this.corePoints = corePoints;
        this.clusterLabels = clusterLabels;
        this.epsilon = epsilon;
        this.distanceMetric = distanceMetric;
        this.strictEpsilonCheck = strictEpsilonCheck;
    }

    public int numClusters() {
        return (int) Arrays.stream(clusterLabels).distinct().count();
    }

    public double[][] corePoints() {
        return corePoints;
    }

    public int[] coreClusterLabels() {
        return clusterLabels;
    }

    @Override
    public int predict(DenseVector features) {
        double[] coordinates = features.values;
        int nearestCoreIdx = -1;
        double minDistance = Double.MAX_VALUE;

        for (int i = 0; i < corePoints.length; i++) {
            double distance = distanceMetric.compute(coordinates, corePoints[i]);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCoreIdx = i;
            }
        }

        boolean isNoise = strictEpsilonCheck && minDistance > epsilon;
        return (nearestCoreIdx < 0 || isNoise) ? -1 : clusterLabels[nearestCoreIdx];
    }
}
