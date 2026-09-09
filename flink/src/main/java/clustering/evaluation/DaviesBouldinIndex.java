package clustering.evaluation;

import clustering.distance.DistanceMetric;

/** Davies-Bouldin index (Davies &amp; Bouldin 1979).
 *
 *  <pre>
 *  DB = (1/k) Σ_i max_{j≠i} (S_i + S_j) / M_ij
 *  </pre>
 *
 *  with {@code S_i} the mean distance of cluster i's points to its centroid and {@code M_ij} the
 *  distance between centroids i and j: each cluster is scored against its worst (most similar)
 *  neighbour, so the index is the average worst-case overlap. <b>Lower is better</b>, 0 is the
 *  floor — the opposite direction to the silhouette and to {@link CalinskiHarabaszIndex}, which
 *  matters when the analysis ranks runs.
 *
 *  Complementary to the silhouette rather than a substitute: it is O(n·d + k²·d) against the
 *  silhouette's O(n²·d), so it runs on the FULL dataset where the silhouette can only be sampled —
 *  but it compares points to centroids only, so it cannot see a cluster's shape and it favours the
 *  convex, isotropic clusters that centroid methods produce anyway. */
public final class DaviesBouldinIndex {

    private DaviesBouldinIndex() {}

    /** The index, from moments already computed. Driver-side and O(k²·d): the whole distributed
     *  part of the work is in {@link ClusterMoments#compute}, which is why the two indices share it.
     *
     *  Degenerate cases follow scikit-learn's {@code davies_bouldin_score} — and the Spark
     *  implementation, so the two engines' columns agree on the degenerate labellings too:
     *  <ul>
     *    <li>fewer than two clusters — the index is undefined (there is no "other" cluster to be
     *        similar to) and 0.0 is reported;</li>
     *    <li>two clusters with COINCIDENT centroids ({@code M_ij = 0}) — the pair is skipped
     *        instead of contributing an infinity, so a duplicate centroid does not turn the whole
     *        run's number into {@code Infinity} (which is not even valid JSON). A cluster whose
     *        every neighbour is skipped scores 0, and if that holds for all of them the index is
     *        0 — optimistic, but it is the reference behaviour and it only arises on degenerate
     *        labellings.</li>
     *  </ul> */
    public static double of(ClusterMoments moments, DistanceMetric distance) {
        ClusterMoments.ClusterMoment[] clusters = moments.clusterMoments;
        int numClusters = clusters.length;
        if (numClusters < 2) {
            return 0.0;
        }

        double[] dispersions = new double[numClusters];
        double[][] centroids = new double[numClusters][];
        for (int i = 0; i < numClusters; i++) {
            dispersions[i] = clusters[i].meanDistance();
            centroids[i] = clusters[i].centroid;
        }

        // Centroid distance matrix, upper triangle only — d is symmetric and the diagonal unused.
        double[][] centroidDistances = new double[numClusters][numClusters];
        for (int i = 0; i < numClusters; i++) {
            for (int j = i + 1; j < numClusters; j++) {
                double d = distance.compute(centroids[i], centroids[j]);
                centroidDistances[i][j] = d;
                centroidDistances[j][i] = d;
            }
        }

        double totalWorstCaseOverlap = 0.0;
        for (int i = 0; i < numClusters; i++) {
            double maxSimilarityRatio = 0.0;
            for (int j = 0; j < numClusters; j++) {
                if (j == i) {
                    continue;
                }
                double d = centroidDistances[i][j];
                if (d > 0.0) {
                    double ratio = (dispersions[i] + dispersions[j]) / d;
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
