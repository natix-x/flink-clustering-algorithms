package clustering.benchmark.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.HashMap;
import java.util.Map;

/** {@code dataset} block of a run config: type + free-form params. */
@JsonIgnoreProperties(ignoreUnknown = true)
public class DataSourceSpec {
    public String type;
    public Map<String, Object> params = new HashMap<>();
}