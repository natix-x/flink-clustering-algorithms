package clustering.benchmark.registry;

import clustering.algorithms.dbscan.CandidateSelectionStrategy;
import clustering.algorithms.dbscan.DBSCANpp;
import clustering.algorithms.dbscan.UniformSelection;
import clustering.algorithms.kmeans.BisectingKMeans;
import clustering.algorithms.kmeans.BreathingKMeans;
import clustering.algorithms.kmeans.KMeans;
import clustering.algorithms.kmedoids.CLARA;
import clustering.algorithms.kmedoids.ClaraFlip;
import clustering.algorithms.kmedoids.DistributedFastPAM;
import clustering.algorithms.kmedoids.DistributedPAM;
import clustering.algorithms.kmedoids.FastPAM;
import clustering.algorithms.kmedoids.PAM;
import clustering.benchmark.config.AlgorithmSpec;
import clustering.benchmark.config.Params;
import clustering.core.Clusterer;
import clustering.core.Geometry;
import clustering.distance.DistanceMetric;
import clustering.distance.EuclideanDistance;

import java.util.Map;
import java.util.function.Function;

/** Name -&gt; Clusterer factory. Mirrors the Spark {@code AlgorithmRegistry}; resolution
 *  and the "unknown algorithm" error come from the shared {@link NamedRegistry}. */
public final class AlgorithmRegistry {

    private AlgorithmRegistry() {}

    /** A built clusterer together with the distance metric it was configured with. Returned by
     *  {@link #create} so evaluation reuses the exact same metric instead of re-parsing the
     *  config — which matters for the k-means family, whose metric comes from {@code geometry}
     *  rather than from a {@code distance} param. Mirrors the Spark {@code Built}. */
    public static final class Built {
        public final Clusterer clusterer;
        public final DistanceMetric distance;

        Built(Clusterer clusterer, DistanceMetric distance) {
            this.clusterer = clusterer;
            this.distance = distance;
        }
    }

    private static final NamedRegistry<Function<Map<String, Object>, Built>> REGISTRY =
        new NamedRegistry<>("algorithm", build());

    private static Map<String, Function<Map<String, Object>, Built>> build() {
        Map<String, Function<Map<String, Object>, Built>> m = new java.util.LinkedHashMap<>();
        m.put("kmeans", AlgorithmRegistry::kmeans);
        m.put("bisectingkmeans", AlgorithmRegistry::bisectingkmeans);
        m.put("pam", AlgorithmRegistry::pam);
        m.put("fastpam", AlgorithmRegistry::fastpam);
        m.put("distfastpam", AlgorithmRegistry::distfastpam);
        m.put("distpam", AlgorithmRegistry::distpam);
        m.put("clara", AlgorithmRegistry::clara);
        m.put("claraflip", AlgorithmRegistry::claraflip);
        m.put("dbscanpp", AlgorithmRegistry::dbscanpp);
        m.put("dbscanexact", AlgorithmRegistry::dbscanexact);
        return m;
    }

    /** K-means on the Flink ML bounded-iteration framework.
     *
     *  {@code geometry: euclidean | spherical} picks the space the centroid update happens in,
     *  and with it the metric — the mean is only geometry-consistent under its own geometry's
     *  metric, so a {@code distance} param is deliberately NOT read here.
     *
     *  {@code refine: none | breathing} is a knob on the same algorithm: {@code breathing}
     *  swaps the plain Lloyd run for Fritzke's add/remove cycle around the identical loop, with
     *  {@code m0} as the breath size. */
    private static Built kmeans(Map<String, Object> params) {
        int k = Params.intParam(params, "k");
        int maxIter = Params.intParam(params, "maxIter", 100);
        double eps = Params.doubleParam(params, "eps", 1e-4);
        long seed = Params.longParam(params, "seed", 42L);
        Geometry geometry = geometryFrom(params);

        String refine = Params.stringParam(params, "refine", "none").toLowerCase();
        Clusterer clusterer;
        switch (refine) {
            case "none":
                clusterer = new KMeans(k, maxIter, eps, seed, geometry);
                break;
            case "breathing":
                clusterer = new BreathingKMeans(
                    k,
                    Params.intParam(params, "m0", 5),
                    maxIter,
                    eps,
                    seed,
                    geometry,
                    Params.intParam(params, "maxCycles", 100));
                break;
            default:
                throw new IllegalArgumentException(
                    "Unknown refine mode: '" + refine + "'. Known: breathing, none");
        }
        return new Built(clusterer, geometry.modelDistance());
    }

    /** Bisecting k-means — divisive, one FLIP-176 job for the whole split search. */
    private static Built bisectingkmeans(Map<String, Object> params) {
        Geometry geometry = geometryFrom(params);
        return new Built(new BisectingKMeans(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 20),
            Params.doubleParam(params, "eps", 1e-4),
            Params.longParam(params, "seed", 42L),
            geometry), geometry.modelDistance());
    }

    /** PAM — classic k-medoids, driver-local O(n²). */
    private static Built pam(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new PAM(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 100),
            distance), distance);
    }

    /** FastPAM — O(k*n²) swap variant, driver-local. */
    private static Built fastpam(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new FastPAM(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 100),
            distance), distance);
    }

    /** Distributed FastPAM — O(n²) work split across machines, no driver-side matrix. */
    private static Built distfastpam(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new DistributedFastPAM(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 100),
            distance), distance);
    }

    /** Distributed classic PAM — same scaffold as distfastpam but the naive O(k*n²) swap.
     *  Baseline for measuring the FastPAM1 speedup. */
    private static Built distpam(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new DistributedPAM(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 100),
            distance), distance);
    }

    /** CLARA — PAM on random samples, scales to large datasets. */
    private static Built clara(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new CLARA(
            Params.intParam(params, "k"),
            Params.intParam(params, "numSamples", 5),
            Params.intParam(params, "sampleSize", 1000),
            Params.intParam(params, "maxIter", 100),
            distance), distance);
    }

    /** CLARA on FLIP-176 — one job, full dataset cached once; PAM stays local on the sample. */
    private static Built claraflip(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new ClaraFlip(
            Params.intParam(params, "k"),
            Params.intParam(params, "numSamples", 5),
            Params.intParam(params, "sampleSize", 1000),
            Params.intParam(params, "maxIter", 100),
            distance), distance);
    }

    /** DBSCAN++ — sampled candidate cores, exact densities against the full dataset. */
    private static Built dbscanpp(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new DBSCANpp(
            Params.doubleParam(params, "eps"),
            Params.intParam(params, "minPts"),
            // The universal accuracy-vs-cost knob; required, so no run hides which m it used.
            Params.doubleParam(params, "coreSampleFraction"),
            CandidateSelectionStrategy.fromName(
                Params.stringParam(params, "sampling", "uniform"),
                Params.intParam(params, "poolFactor", 4)),
            // assign: 'eps' = classic DBSCAN noise semantics, 'closest' = the paper's rule.
            assignWithinEps(Params.stringParam(params, "assign", "eps")),
            Params.intParam(params, "chunkSize", 2000),
            distance,
            Params.longParam(params, "seed", 42L)), distance);
    }

    /** Exact classic DBSCAN — the SAME code path as {@code dbscanpp} with s pinned to 1.0, so
     *  densities, core points and connectivity are all exact. Registered under its own name so
     *  the experiment matrix and the thesis tables can carry the exact rung as a distinct row
     *  rather than as a parameter value of the sampled entry.
     *
     *  O(n²) time and every point resident as a candidate: the exactness ORACLE, for samples and
     *  small data, not for the full datasets. */
    private static Built dbscanexact(Map<String, Object> params) {
        double s = Params.doubleParam(params, "coreSampleFraction", 1.0);
        if (s != 1.0) {
            throw new IllegalArgumentException(
                "'dbscanexact' is exact by definition (coreSampleFraction = 1.0), got " + s
                + ". Use 'dbscanpp' for sampled runs.");
        }
        // At s = 1.0 every strategy degenerates to "take all rows", so accepting these silently
        // would leave a config claiming a sampling strategy that never ran.
        for (String key : new String[] {"sampling", "poolFactor"}) {
            if (params.containsKey(key)) {
                throw new IllegalArgumentException(
                    "'dbscanexact' takes no candidate sampling params (got '" + key + "'): with "
                    + "coreSampleFraction = 1.0 every candidate is used. Use 'dbscanpp' to sample.");
            }
        }
        DistanceMetric distance = distanceFrom(params);
        return new Built(new DBSCANpp(
            Params.doubleParam(params, "eps"),
            Params.intParam(params, "minPts"),
            1.0,
            UniformSelection.INSTANCE,
            assignWithinEps(Params.stringParam(params, "assign", "eps")),
            Params.intParam(params, "chunkSize", 2000),
            distance,
            Params.longParam(params, "seed", 42L)), distance);
    }

    public static Built create(AlgorithmSpec spec) {
        Function<Map<String, Object>, Built> f = REGISTRY.get(spec.name);
        return f.apply(spec.params == null ? new java.util.HashMap<>() : spec.params);
    }

    private static DistanceMetric distanceFrom(Map<String, Object> params) {
        Object d = params.get("distance");
        return d == null ? EuclideanDistance.INSTANCE : DistanceRegistry.get(d.toString());
    }

    /** Parses the optional {@code geometry} param (the space a centroid algorithm optimises in).
     *  Absent = euclidean, so existing configs keep their meaning. */
    private static Geometry geometryFrom(Map<String, Object> params) {
        Object g = params.get("geometry");
        return g == null ? GeometryRegistry.DEFAULT : GeometryRegistry.get(g.toString());
    }

    private static boolean assignWithinEps(String assign) {
        switch (assign.toLowerCase()) {
            case "eps":
                return true;
            case "closest":
                return false;
            default:
                throw new IllegalArgumentException(
                    "Unknown assign mode: '" + assign + "'. Known: closest, eps");
        }
    }
}
