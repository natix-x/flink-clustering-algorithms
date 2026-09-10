package clustering.benchmark.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Arrays;
import java.util.List;

/**
 * Configuration for evaluation metrics and sampling limits.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class EvaluationSpec {

    public static final int DEFAULT_SAMPLE_SIZE = 10_000;

    public List<String> metrics = Arrays.asList("silhouette", "nClusters", "clusterSizes", "noiseFraction");

    /**
     * Sample size for expensive metrics (e.g., silhouette).
     * An explicit null indicates no sampling (evaluates on the full dataset).
     */
    private Integer sampleSize = DEFAULT_SAMPLE_SIZE;

    public long seed = 42L;

    @JsonProperty("sampleSize")
    public void setSampleSize(Integer sampleSize) {
        this.sampleSize = sampleSize;
    }

    @JsonProperty("sampleSize")
    public Integer sampleSize() {
        return sampleSize;
    }

    public boolean wants(String metric) {
        return metrics != null && metrics.contains(metric);
    }
}
