package clustering.benchmark.evaluation;

import java.util.Map;


/** Evaluation outputs. Every field is optional: only the metrics named in
 *  {@code EvaluationSpec.metrics} are computed, the rest stay null (and are omitted from the
 *  emitted {@code RunResult}). Field-for-field mirror of the Spark {@code EvaluationResult}. */
public final class EvaluationResult {

    public final Integer nClusters;
    public final Double noiseFraction;
    public final Double silhouette;
    /** Drawn points the silhouette actually scored — the achieved sample size, which is Binomial
     *  around {@code sampleSize} rather than equal to it. */
    public final Integer silhouetteScoredPoints;
    /** Distinct clusters present in the silhouette's sample. Below {@code nClusters} means whole
     *  clusters missed the draw, so {@code b} was minimised over the survivors only and the score
     *  reads HIGH. */
    public final Integer silhouetteSampleClusters;
    /** Drawn points the silhouette refused: a non-finite coordinate or weight, or a cluster that
     *  drew fewer than two sample members. A large value means the score describes a noticeably
     *  smaller population than the draw intended. */
    public final Integer silhouetteUnscoredPoints;
    public final Map<String, Long> clusterSizes;
    public final Double daviesBouldin;
    public final Double calinskiHarabasz;

    public EvaluationResult(Integer nClusters, Double noiseFraction, Double silhouette,
                            Integer silhouetteScoredPoints, Integer silhouetteSampleClusters,
                            Integer silhouetteUnscoredPoints, Map<String, Long> clusterSizes,
                            Double daviesBouldin, Double calinskiHarabasz) {
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

    /** No metrics computed — used for a failed run. */
    public static final EvaluationResult EMPTY =
        new EvaluationResult(null, null, null, null, null, null, null, null, null);
}
