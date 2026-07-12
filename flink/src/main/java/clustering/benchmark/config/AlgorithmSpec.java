package clustering.benchmark.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.HashMap;
import java.util.Map;


@JsonIgnoreProperties(ignoreUnknown = true)
public class AlgorithmSpec {
    public String name;
    public Map<String, Object> params = new HashMap<>();
}
