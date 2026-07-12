package clustering.benchmark.evaluation;

import java.util.Map;


public final class EvaluationResult {

    public final Integer nClusters;
    public final Double noiseFraction;
    public final Double silhouette;
    public final Map<String, Long> clusterSizes;

    public EvaluationResult(Integer nClusters, Double noiseFraction, Double silhouette,
                            Map<String, Long> clusterSizes) {
        this.nClusters = nClusters;
        this.noiseFraction = noiseFraction;
        this.silhouette = silhouette;
        this.clusterSizes = clusterSizes;
    }

    /** No metrics computed — used for a failed run. */
    public static final EvaluationResult EMPTY = new EvaluationResult(null, null, null, null);
}
