package clustering.evaluation;

import clustering.distance.DistanceMetric;

/**
 * Computes the Calinski-Harabasz Index (Variance Ratio Criterion).
 * Higher values indicate better-defined clusters.
 */
public final class CalinskiHarabaszIndex {

    private CalinskiHarabaszIndex() {}

    /**
     * Computes the index from pre-calculated cluster moments.
     *
     * Edge cases:
     * - Returns 0.0 if there are fewer than 2 clusters or calculations result in non-finite values.
     * - Returns 1.0 if within-cluster dispersion is zero (e.g., all points sit exactly on their centroids).
     */
    public static double compute(ClusterMoments moments, DistanceMetric distanceMetric) {
        ClusterMoments.ClusterMoment[] clusterMoments = moments.clusterMoments;
        int numClusters = clusterMoments.length;
        double totalDataWeight = moments.totalWeight();

        if (numClusters < 2 || totalDataWeight <= numClusters) {
            return 0.0;
        }

        double withinClusterDispersion = 0.0;
        for (ClusterMoments.ClusterMoment cluster : clusterMoments) {
            withinClusterDispersion += cluster.sumOfWeightedSquaredDistances;
        }

        if (!Double.isFinite(withinClusterDispersion) || withinClusterDispersion <= 0.0) {
            return 1.0;
        }

        double betweenClusterDispersion = 0.0;
        for (ClusterMoments.ClusterMoment cluster : clusterMoments) {
            double distanceToGlobalCentroid = distanceMetric.compute(cluster.centroid, moments.globalCentroid);
            betweenClusterDispersion += cluster.clusterWeight * distanceToGlobalCentroid * distanceToGlobalCentroid;
        }

        double betweenClusterVariance = betweenClusterDispersion / (numClusters - 1);
        double withinClusterVariance = withinClusterDispersion / (totalDataWeight - numClusters);

        double index = betweenClusterVariance / withinClusterVariance;
        return Double.isFinite(index) ? index : 0.0;
    }
}
