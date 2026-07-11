package clustering.benchmark.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/** Root config for a single benchmark run. Java mirror of the Spark
 *  {@code clustering.benchmark.config.RunConfig}, parsed with Jackson and
 *  conforming to contract/run_config.schema.json.
 *
 *  Accepts both the neutral {@code engineConf} and the deprecated Spark alias
 *  {@code sparkConf}; {@link #effectiveEngineConf()} merges them. */
@JsonIgnoreProperties(ignoreUnknown = true)
public class RunConfig {

    public String runId;
    public String profile;            // nullable -> env auto-detect
    public DataSourceSpec dataset;
    public AlgorithmSpec algorithm;
    public EvaluationSpec evaluation = new EvaluationSpec();
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

    public ClusterProfile resolveProfile() {
        return profile != null ? ClusterProfile.fromName(profile) : ClusterProfile.fromEnv();
    }

    public String resolveOutputDir(ClusterProfile resolved) {
        return outputDir != null ? outputDir : resolved.outputDir();
    }
}