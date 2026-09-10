package clustering.benchmark.evaluation;

import java.util.Map;

/**
 * Container for evaluation outputs.
 * Fields are optional and remain null if the corresponding metric was not requested.
 */
public final class EvaluationResult {

    public final Integer nClusters;
    public final Double noiseFraction;
    public final Double silhouette;

    /** Number of sampled points successfully scored by the silhouette metric. */
    public final Integer silhouetteScoredPoints;

    /** Number of distinct clusters actually present in the silhouette's sample. */
    public final Integer silhouetteSampleClusters;

    /** Number of drawn points excluded from the silhouette score (e.g., due to invalid data or singleton clusters). */
    public final Integer silhouetteUnscoredPoints;

    public final Map<String, Long> clusterSizes;
    public final Double daviesBouldin;
    public final Double calinskiHarabasz;

    public EvaluationResult(
            Integer nClusters,
            Double noiseFraction,
            Double silhouette,
            Integer silhouetteScoredPoints,
            Integer silhouetteSampleClusters,
            Integer silhouetteUnscoredPoints,
            Map<String, Long> clusterSizes,
            Double daviesBouldin,
            Double calinskiHarabasz
    ) {
        this.nClusters = nClusters;
        this.noiseFraction = noiseFraction;
        this.silhouette = silhouette;
        this.silhouetteScoredPoints = silhouetteScoredPoints;
        this.silhouetteSampleClusters = silhouetteSampleClusters;
        this.silhouetteUnscoredPoints = silhouetteUnscoredPoints;
        this.clusterSizes = clusterSizes;
        this.daviesBouldin = daviesBouldin;
        this.calinskiHarabasz = calinskiHarabasz;
    }

    /** Empty result instance used for failed runs or when no metrics are computed. */
    public static final EvaluationResult EMPTY =
        new EvaluationResult(null, null, null, null, null, null, null, null, null);
}
