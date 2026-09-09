package clustering.benchmark.config;


import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


@JsonIgnoreProperties(ignoreUnknown = true)
public class RunConfig {

    public String runId;
    public String profile;            // nullable -> env auto-detect
    public DataSourceSpec dataset;
    public AlgorithmSpec algorithm;
    public EvaluationSpec evaluation = new EvaluationSpec();
    /** Per-run engine configuration. The harness emits this under the key `flink_config`
     *  (see the run-config contract and `flink_launcher.py`), which the alias below binds —
     *  without it, `@JsonIgnoreProperties(ignoreUnknown = true)` silently DROPPED every
     *  entry, so the knob looked live while doing nothing, unlike Spark's working
     *  `spark_config`. Applied to the execution environment in `FlinkClusteringJob`. */
    @JsonAlias({"flink_config", "flinkConf"})
    public Map<String, String> engineConf = new HashMap<>();
    public Map<String, String> sparkConf;   // deprecated alias
    public String outputDir;          // overrides profile default
    public Map<String, String> experimentMetadata = new HashMap<>();

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static RunConfig fromFile(String path) throws IOException {
        return MAPPER.readValue(new File(path), RunConfig.class);
    }

    public static RunConfig fromJsonString(String json) throws IOException {
        return MAPPER.readValue(json, RunConfig.class);
    }

    /** engineConf with the deprecated sparkConf alias merged in (engineConf wins). */
    public Map<String, String> effectiveEngineConf() {
        Map<String, String> merged = new HashMap<>();
        if (sparkConf != null) merged.putAll(sparkConf);
        if (engineConf != null) merged.putAll(engineConf);
        return merged;
    }

    /** Local-dev default output dir; the harness overrides it via the config's outputDir.
     *  Single default, profile-independent — mirrors the Spark repo's RunConfig. */
    private static final String DEFAULT_OUTPUT_DIR =
        System.getProperty("user.dir", ".") + "/benchmark-results";

    /** profile name -> ClusterProfile; absent -> local (mirrors Spark's getOrElse(LocalProfile)). */
    public ClusterProfile resolveProfile() {
        return profile != null ? ClusterProfile.fromName(profile) : new ClusterProfile.LocalProfile();
    }

    public String resolveOutputDir() {
        return outputDir != null ? outputDir : DEFAULT_OUTPUT_DIR;
    }
}
