package clustering.benchmark.registry;

import clustering.benchmark.config.DataSourceSpec;
import clustering.benchmark.datasource.DataSource;
import clustering.benchmark.datasource.SyntheticDataSource;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/** Maps a {@code DataSourceSpec.type} to its {@link DataSource.Factory}. Mirrors the Spark
 *  {@code DataSourceRegistry}; resolution and the "unknown type" error come from the shared
 *  {@link NamedRegistry}, so all registration lives in one package alongside
 *  {@link AlgorithmRegistry} and {@link DistanceRegistry}. Add a data source by listing its
 *  Factory here (e.g. {@code ParquetDataSource.factory()}). */
public final class DataSourceRegistry {

    private static final NamedRegistry<DataSource.Factory> REGISTRY;

    static {
        Map<String, DataSource.Factory> m = new LinkedHashMap<>();
        register(m, SyntheticDataSource.factory());
        // future: register(m, ParquetDataSource.factory());
        REGISTRY = new NamedRegistry<>("data source type", m);
    }

    private DataSourceRegistry() {}

    private static void register(Map<String, DataSource.Factory> m, DataSource.Factory f) {
        m.put(f.typeName(), f);
    }

    public static DataSource create(DataSourceSpec spec) {
        return REGISTRY.get(spec.type).create(spec.params == null ? new HashMap<>() : spec.params);
    }
}
