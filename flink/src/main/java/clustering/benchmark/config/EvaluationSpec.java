package clustering.benchmark.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.Arrays;
import java.util.List;


@JsonIgnoreProperties(ignoreUnknown = true)
public class EvaluationSpec {
    public List<String> metrics = Arrays.asList("silhouette", "nClusters", "clusterSizes", "noiseFraction");
    public Integer sampleSize;   // null = full dataset
    public long seed = 42L;

    public boolean wants(String metric) {
        return metrics != null && metrics.contains(metric);
    }
}
