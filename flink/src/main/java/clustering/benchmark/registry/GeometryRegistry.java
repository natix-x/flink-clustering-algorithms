package clustering.benchmark.registry;

import clustering.core.EuclideanGeometry;
import clustering.core.Geometry;
import clustering.core.SphericalGeometry;
import java.util.LinkedHashMap;
import java.util.Map;

/** Maps the {@code geometry} param to a {@link Geometry} — mirrors {@link DistanceRegistry} so
 *  every config string is resolved the same way, with the same "unknown X" error. */
public final class GeometryRegistry {

    /** Used when a config omits {@code geometry}, keeping old configs' behaviour unchanged. */
    public static final Geometry DEFAULT = EuclideanGeometry.INSTANCE;

    private static final NamedRegistry<Geometry> REGISTRY;

    static {
        Map<String, Geometry> m = new LinkedHashMap<>();
        m.put(EuclideanGeometry.INSTANCE.name(), EuclideanGeometry.INSTANCE);
        m.put(SphericalGeometry.INSTANCE.name(), SphericalGeometry.INSTANCE);
        REGISTRY = new NamedRegistry<>("geometry", m);
    }

    private GeometryRegistry() {}

    public static Geometry get(String name) {
        return REGISTRY.get(name);
    }
}
