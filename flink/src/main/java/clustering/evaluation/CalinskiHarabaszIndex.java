package clustering.evaluation;

import clustering.distance.DistanceMetric;

/** Calinski-Harabasz index, a.k.a. the variance ratio criterion (Calinski &amp; Harabasz 1974).
 *
 *  <pre>
 *  CH = [ Σ_i n_i·d(c_i, c)² / (k-1) ] / [ Σ_i Σ_{x∈C_i} d(x, c_i)² / (n-k) ]
 *  </pre>
 *
 *  — between-cluster dispersion over within-cluster dispersion, each divided by its degrees of
 *  freedom. <b>Higher is better</b>, and it is unbounded above, so the value only means something
 *  compared against another labelling OF THE SAME DATA: unlike the silhouette it is not normalised
 *  to a fixed range and cannot be compared across datasets.
 *
 *  With the Euclidean metric the two dispersions are the usual between/within sums of squares and
 *  the index is the classical F-like ratio; with another metric it is the same formula on that
 *  metric's squared distances, which keeps the number consistent with the objective the run's
 *  algorithm actually optimised (a k-medoids run under {@code manhattan} is scored under
 *  {@code manhattan}) at the cost of the ANOVA reading.
 *
 *  Like {@link DaviesBouldinIndex} it is O(n·d) and therefore computed on the FULL dataset, never
 *  a sample. */
public final class CalinskiHarabaszIndex {

    private CalinskiHarabaszIndex() {}

    /** The index, from moments already computed — driver-side arithmetic over k clusters.
     *
     *  {@code totalDataWeight} is the total WEIGHT of the non-noise points, not the row count, so
     *  weighting stays equivalent to duplication; on unweighted input the two are the same number.
     *
     *  Degenerate cases follow scikit-learn's {@code calinski_harabasz_score}, as the Spark side
     *  does:
     *  <ul>
     *    <li>fewer than two clusters, or {@code totalDataWeight <= numClusters} (no within-cluster
     *        degrees of freedom) — undefined, reported as 0.0, the convention every evaluator here
     *        uses for "could not be computed";</li>
     *    <li>zero within-cluster dispersion (every point sits exactly on its centroid) — 1.0,
     *        since the ratio's denominator vanishes.</li>
     *  </ul>
     *
     *  {@code isFinite} catches NaN as well as infinity: a NaN coordinate propagates through the
     *  moments, and NaN fails every {@code <=} test, so it would otherwise reach the emitted result
     *  — where it serialises as a bare {@code NaN} token that is not valid JSON, losing the whole
     *  run's file rather than one number. */
    public static double of(ClusterMoments moments, DistanceMetric distance) {
        ClusterMoments.ClusterMoment[] clusters = moments.clusterMoments;
        int numClusters = clusters.length;
        double totalDataWeight = moments.totalWeight();

        if (numClusters < 2 || totalDataWeight <= numClusters) {
            return 0.0;
        }

        double withinClusterDispersion = 0.0;
        for (ClusterMoments.ClusterMoment cluster : clusters) {
            withinClusterDispersion += cluster.sumOfWeightedSquaredDistances;
        }
        if (!Double.isFinite(withinClusterDispersion) || withinClusterDispersion <= 0.0) {
            return 1.0;
        }

        double betweenClusterDispersion = 0.0;
        for (ClusterMoments.ClusterMoment cluster : clusters) {
            double d = distance.compute(cluster.centroid, moments.globalCentroid);
            betweenClusterDispersion += cluster.clusterWeight * d * d;
        }

        double betweenClusterVariance = betweenClusterDispersion / (numClusters - 1);
        double withinClusterVariance = withinClusterDispersion / (totalDataWeight - numClusters);

        double index = betweenClusterVariance / withinClusterVariance;
        return Double.isFinite(index) ? index : 0.0;
    }
}
