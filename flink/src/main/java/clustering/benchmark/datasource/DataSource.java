package clustering.benchmark.datasource;

import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.Map;

/** Source of points ({@link DenseVector}) for one run. Java mirror of the Spark
 *  {@code clustering.benchmark.datasource.DataSource}. Implementations live in this
 *  package and register their {@link Factory} with
 *  {@code clustering.benchmark.registry.DataSourceRegistry}. */
public interface DataSource {

    String name();

    DataStream<DenseVector> load(StreamExecutionEnvironment env);

    Map<String, String> metadata();

    /** Builds a {@link DataSource} from its config params. Add a source by defining a
     *  Factory and listing it in {@code DataSourceRegistry}. */
    interface Factory {
        String typeName();
        DataSource create(Map<String, Object> params);
    }
}
