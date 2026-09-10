package clustering.evaluation;

import clustering.distance.DistanceMetric;

/**
 * Computes the Davies-Bouldin Index.
 *
 * $$DB=\frac{1}{k}\sum_{i}\max_{j\neq i}\frac{S_i+S_j}{M_{ij}}$$
 *
 * Where $S_i$ is the mean distance of cluster points to their centroid,
 * and $M_{ij}$ is the distance between centroids $i$ and $j$.
 * Lower values indicate better clustering (0.0 is the minimum).
 */
public final class DaviesBouldinIndex {

    private DaviesBouldinIndex() {}

    /**
     * Computes the index from pre-calculated cluster moments.
     *
     * Edge cases:
     * - Returns 0.0 if there are fewer than 2 clusters.
     * - Coincident centroids (distance of 0.0) are skipped to prevent division by zero.
     */
    public static double compute(ClusterMoments moments, DistanceMetric distanceMetric) {
        ClusterMoments.ClusterMoment[] clusterMoments = moments.clusterMoments;
        int numClusters = clusterMoments.length;

        if (numClusters < 2) {
            return 0.0;
        }

        double[] dispersions = new double[numClusters];
        double[][] centroids = new double[numClusters][];
        for (int i = 0; i < numClusters; i++) {
            dispersions[i] = clusterMoments[i].meanDistance();
            centroids[i] = clusterMoments[i].centroid;
        }

        // Centroid distance matrix, upper triangle only
        double[][] interCentroidDistances = new double[numClusters][numClusters];
        for (int i = 0; i < numClusters; i++) {
            for (int j = i + 1; j < numClusters; j++) {
                double distance = distanceMetric.compute(centroids[i], centroids[j]);
                interCentroidDistances[i][j] = distance;
                interCentroidDistances[j][i] = distance;
            }
        }

        double totalWorstCaseOverlap = 0.0;
        for (int i = 0; i < numClusters; i++) {
            double maxSimilarityRatio = 0.0;
            for (int j = 0; j < numClusters; j++) {
                if (j == i) {
                    continue;
                }

                double centroidDistance = interCentroidDistances[i][j];
                if (centroidDistance > 0.0) {
                    double ratio = (dispersions[i] + dispersions[j]) / centroidDistance;
                    if (ratio > maxSimilarityRatio) {
                        maxSimilarityRatio = ratio;
                    }
                }
            }
            totalWorstCaseOverlap += maxSimilarityRatio;
        }

        return totalWorstCaseOverlap / numClusters;
    }
}
