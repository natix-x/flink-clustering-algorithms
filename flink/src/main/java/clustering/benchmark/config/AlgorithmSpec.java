package clustering.benchmark.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.HashMap;
import java.util.Map;

/** {@code algorithm} block of a run config: name + free-form params. */
@JsonIgnoreProperties(ignoreUnknown = true)
public class AlgorithmSpec {
    public String name;
    public Map<String, Object> params = new HashMap<>();
}