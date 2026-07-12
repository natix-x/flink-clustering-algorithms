package clustering.benchmark.registry;

import clustering.distance.CosineDistance;
import clustering.distance.DistanceMetric;
import clustering.distance.EuclideanDistance;
import clustering.distance.ManhattanDistance;

import java.util.LinkedHashMap;
import java.util.Map;

/** Name -> distance metric. Mirrors the Spark {@code DistanceRegistry}; resolution
 *  and the "unknown metric" error come from the shared {@link NamedRegistry}. */
public final class DistanceRegistry {

    private static final NamedRegistry<DistanceMetric> REGISTRY;

    static {
        Map<String, DistanceMetric> m = new LinkedHashMap<>();
        m.put("euclidean", EuclideanDistance.INSTANCE);
        m.put("manhattan", ManhattanDistance.INSTANCE);
        m.put("cosine", CosineDistance.INSTANCE);
        REGISTRY = new NamedRegistry<>("distance metric", m);
    }

    private DistanceRegistry() {}

    public static DistanceMetric get(String name) {
        return REGISTRY.get(name);
    }
}
