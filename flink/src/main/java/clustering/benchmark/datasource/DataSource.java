package clustering.benchmark.datasource;

import clustering.benchmark.config.DataSourceSpec;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeSet;

/** Source of points ({@code double[]}) for one run. Java mirror of the Spark
 *  {@code clustering.benchmark.datasource.DataSource} with a built-in registry. */
public interface DataSource {

    String name();

    DataStream<double[]> load(StreamExecutionEnvironment env);

    Map<String, String> metadata();

    interface Factory {
        String typeName();
        DataSource create(Map<String, Object> params);
    }

    // --- registry ---------------------------------------------------------
    Map<String, Factory> FACTORIES = Registry.builtins();

    final class Registry {
        private Registry() {}

        static Map<String, Factory> builtins() {
            Map<String, Factory> m = new HashMap<>();
            Factory synth = SyntheticDataSource.factory();
            m.put(synth.typeName().toLowerCase(), synth);
            return m;
        }
    }

    static DataSource create(DataSourceSpec spec) {
        Factory f = FACTORIES.get(spec.type.toLowerCase());
        if (f == null) {
            throw new IllegalArgumentException(
                "Unknown data source type: '" + spec.type + "'. Known: "
                    + new TreeSet<>(FACTORIES.keySet()));
        }
        return f.create(spec.params == null ? new HashMap<>() : spec.params);
    }
}