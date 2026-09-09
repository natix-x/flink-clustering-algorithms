package clustering.algorithms.kmedoids.components;

import clustering.algorithms.kmedoids.local.FastPAM;
import clustering.algorithms.kmedoids.local.FasterPAM;
import clustering.distance.DistanceMetric;

/**
 * Dense pairwise distance matrix backed by a flat row-major array.
 * Optimized for cache efficiency in local k-medoids algorithms (e.g., FastPAM, FasterPAM).
 */
public final class DistanceMatrix {

    private final double[] distances;
    public final int numPoints;

    private DistanceMatrix(double[] distances, int numPoints) {
        this.distances = distances;
        this.numPoints = numPoints;
    }

    public double getDistance(int rowIndex, int colIndex) {
        return distances[rowIndex * numPoints + colIndex];
    }

    /**
     * Computes the symmetric pairwise distance matrix.
     * Evaluates only the upper triangle to minimize distance computations.
     */
    public static DistanceMatrix computePairwise(double[][] points, DistanceMetric distanceMetric) {
        int numPoints = points.length;
        double[] distances = new double[numPoints * numPoints];

        for (int i = 0; i < numPoints; i++) {
            for (int j = i + 1; j < numPoints; j++) {
                double dist = distanceMetric.compute(points[i], points[j]);
                distances[i * numPoints + j] = dist;
                distances[j * numPoints + i] = dist;
            }
        }
        return new DistanceMatrix(distances, numPoints);
    }
}
