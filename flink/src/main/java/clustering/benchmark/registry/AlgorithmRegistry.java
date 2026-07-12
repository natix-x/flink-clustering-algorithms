package clustering.benchmark.registry;

import clustering.algorithms.kmeans.FlinkMLKMeans;
import clustering.algorithms.kmedoids.CLARA;
import clustering.algorithms.kmedoids.ClaraFlip;
import clustering.algorithms.kmedoids.DistributedFastPAM;
import clustering.algorithms.kmedoids.DistributedPAM;
import clustering.algorithms.kmedoids.FastPAM;
import clustering.algorithms.kmedoids.PAM;
import clustering.benchmark.config.AlgorithmSpec;
import clustering.benchmark.config.Params;
import clustering.core.Clusterer;
import clustering.distance.DistanceMetric;
import clustering.distance.EuclideanDistance;

import java.util.Map;
import java.util.function.Function;

/** Name -> Clusterer factory. Mirrors the Spark {@code AlgorithmRegistry}; resolution
 *  and the "unknown algorithm" error come from the shared {@link NamedRegistry}. */
public final class AlgorithmRegistry {

    private AlgorithmRegistry() {}

    private static final NamedRegistry<Function<Map<String, Object>, Clusterer>> REGISTRY =
        new NamedRegistry<>("algorithm", build());

    private static Map<String, Function<Map<String, Object>, Clusterer>> build() {
        Map<String, Function<Map<String, Object>, Clusterer>> m = new java.util.LinkedHashMap<>();
        m.put("kmeans", AlgorithmRegistry::kmeans);
        m.put("pam", AlgorithmRegistry::pam);
        m.put("fastpam", AlgorithmRegistry::fastpam);
        m.put("distfastpam", AlgorithmRegistry::distfastpam);
        m.put("distpam", AlgorithmRegistry::distpam);
        m.put("clara", AlgorithmRegistry::clara);
        m.put("claraflip", AlgorithmRegistry::claraflip);
        return m;
    }

    /** KMeans on the Flink ML bounded-iteration framework. */
    private static Clusterer kmeans(Map<String, Object> params) {
        return new FlinkMLKMeans(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 100),
            Params.doubleParam(params, "eps", 1e-4),
            distanceFrom(params),
            Params.longParam(params, "seed", 42L));
    }

    /** PAM — classic k-medoids, driver-local O(n²). */
    private static Clusterer pam(Map<String, Object> params) {
        return new PAM(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 100),
            distanceFrom(params));
    }

    /** FastPAM — O(k*n²) swap variant, driver-local. */
    private static Clusterer fastpam(Map<String, Object> params) {
        return new FastPAM(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 100),
            distanceFrom(params));
    }

    /** Distributed FastPAM — O(n²) work split across machines, no driver-side matrix. */
    private static Clusterer distfastpam(Map<String, Object> params) {
        return new DistributedFastPAM(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 100),
            distanceFrom(params));
    }

    /** Distributed classic PAM — same scaffold as distfastpam but the naive O(k*n²) swap.
     *  Baseline for measuring the FastPAM1 speedup. */
    private static Clusterer distpam(Map<String, Object> params) {
        return new DistributedPAM(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 100),
            distanceFrom(params));
    }

    /** CLARA — PAM on random samples, scales to large datasets. */
    private static Clusterer clara(Map<String, Object> params) {
        return new CLARA(
            Params.intParam(params, "k"),
            Params.intParam(params, "numSamples", 5),
            Params.intParam(params, "sampleSize", 1000),
            Params.intParam(params, "maxIter", 100),
            distanceFrom(params));
    }

    /** CLARA on FLIP-176 — one job, full dataset cached once; PAM stays local on the sample. */
    private static Clusterer claraflip(Map<String, Object> params) {
        return new ClaraFlip(
            Params.intParam(params, "k"),
            Params.intParam(params, "numSamples", 5),
            Params.intParam(params, "sampleSize", 1000),
            Params.intParam(params, "maxIter", 100),
            distanceFrom(params));
    }

    public static Clusterer create(AlgorithmSpec spec) {
        Function<Map<String, Object>, Clusterer> f = REGISTRY.get(spec.name);
        return f.apply(spec.params == null ? new java.util.HashMap<>() : spec.params);
    }

    private static DistanceMetric distanceFrom(Map<String, Object> params) {
        Object d = params.get("distance");
        return d == null ? EuclideanDistance.INSTANCE : DistanceRegistry.get(d.toString());
    }
}
