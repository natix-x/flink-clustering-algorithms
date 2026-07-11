package clustering.benchmark.registry;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/** Case-insensitive name -> value lookup with a uniform "unknown X" error.
 *  Shared by the algorithm and distance registries so both resolve names and
 *  report unknown ones exactly the same way. Java mirror of the Spark
 *  {@code clustering.benchmark.registry.NamedRegistry}. */
final class NamedRegistry<A> {

    private final String kind;
    private final Map<String, A> byName;

    NamedRegistry(String kind, Map<String, A> entries) {
        this.kind = kind;
        Map<String, A> m = new LinkedHashMap<>();
        for (Map.Entry<String, A> e : entries.entrySet()) {
            m.put(e.getKey().toLowerCase(), e.getValue());
        }
        this.byName = m;
    }

    /** Registered names, sorted — for error messages and discovery. */
    Set<String> knownNames() {
        return new TreeSet<>(byName.keySet());
    }

    A get(String name) {
        A v = byName.get(name.toLowerCase());
        if (v == null) {
            throw new IllegalArgumentException(
                "Unknown " + kind + ": '" + name + "'. Known: " + knownNames());
        }
        return v;
    }
}
