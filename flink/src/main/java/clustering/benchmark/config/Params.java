package clustering.benchmark.config;

import java.util.Map;

/** Typed access to the free-form {@code params} maps (Jackson decodes JSON
 *  numbers as Integer/Long/Double). Mirrors the param helpers in the Spark
 *  registries. */
public final class Params {

    private Params() {}

    public static int intParam(Map<String, Object> p, String key) {
        Object v = require(p, key);
        return ((Number) v).intValue();
    }

    public static int intParam(Map<String, Object> p, String key, int dflt) {
        Object v = p.get(key);
        return v == null ? dflt : ((Number) v).intValue();
    }

    public static long longParam(Map<String, Object> p, String key) {
        return ((Number) require(p, key)).longValue();
    }

    public static long longParam(Map<String, Object> p, String key, long dflt) {
        Object v = p.get(key);
        return v == null ? dflt : ((Number) v).longValue();
    }

    public static double doubleParam(Map<String, Object> p, String key) {
        Object v = require(p, key);
        return ((Number) v).doubleValue();
    }

    public static double doubleParam(Map<String, Object> p, String key, double dflt) {
        Object v = p.get(key);
        return v == null ? dflt : ((Number) v).doubleValue();
    }

    public static String stringParam(Map<String, Object> p, String key, String dflt) {
        Object v = p.get(key);
        return v == null ? dflt : v.toString();
    }

    private static Object require(Map<String, Object> p, String key) {
        Object v = p.get(key);
        if (v == null) {
            throw new IllegalArgumentException("Missing required parameter: '" + key + "'");
        }
        return v;
    }
}