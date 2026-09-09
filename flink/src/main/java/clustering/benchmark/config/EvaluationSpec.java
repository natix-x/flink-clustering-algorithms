package clustering.benchmark.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Arrays;
import java.util.List;


/** Which metrics to compute, and on how much data. Mirrors the Spark {@code EvaluationSpec} —
 *  including the defaults, because a config that omits a key must not make the two engines measure
 *  different things. */
@JsonIgnoreProperties(ignoreUnknown = true)
public class EvaluationSpec {

    /** Same default set as Spark. {@code daviesBouldin} and {@code calinskiHarabasz} are known
     *  metrics but off by default: they cost two extra full passes, which is not something a config
     *  should pay for by omission. */
    public List<String> metrics = Arrays.asList("silhouette", "nClusters", "clusterSizes", "noiseFraction");

    /** Points scored by the silhouette, the one O(n²) metric. Defaults to 10 000, which is ALSO
     *  Spark's default.
     *
     *  <p>An explicit {@code null} means NO sampling: the whole labelled dataset is collected to the
     *  driver, which is an exactness oracle on small data and a driver OOM on Gaia (434 M rows) or
     *  Cohere (113 M × 1024). Ask for it deliberately, never by omission — hence the setter: an
     *  ABSENT key keeps the 10 000 default, an explicit null turns sampling off, and the two are
     *  different requests. Above
     *  {@link clustering.benchmark.evaluation.EvaluationRunner#FULL_SILHOUETTE_MAX_ROWS} labelled
     *  rows it is refused outright, so the omission fails the run at the evaluation boundary instead
     *  of OOM-ing the driver after the fit has already been paid for. */
    private Integer sampleSize = DEFAULT_SAMPLE_SIZE;

    public long seed = 42L;

    public static final int DEFAULT_SAMPLE_SIZE = 10_000;

    @JsonProperty("sampleSize")
    public void setSampleSize(Integer sampleSize) {
        this.sampleSize = sampleSize;
    }

    /** {@code null} = no sampling. */
    @JsonProperty("sampleSize")
    public Integer sampleSize() {
        return sampleSize;
    }

    public boolean wants(String metric) {
        return metrics != null && metrics.contains(metric);
    }
}
