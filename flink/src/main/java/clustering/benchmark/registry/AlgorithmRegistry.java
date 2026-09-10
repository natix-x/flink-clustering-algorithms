package clustering.benchmark.registry;

import clustering.algorithms.dbscan.components.UniformSelection;
import clustering.algorithms.kmeans.hierarchical.BisectingKMeans;
import clustering.algorithms.kmedoids.distributed.DistributedFastPAM;
import clustering.algorithms.kmedoids.hybrid.CLARA;
import clustering.algorithms.kmedoids.hybrid.PAMAE;
import clustering.algorithms.kmedoids.local.FastPAM;
import clustering.algorithms.kmedoids.local.FasterPAM;
import clustering.algorithms.dbscan.DBSCANpp;
import clustering.algorithms.kmeans.BreathingKMeans;
import clustering.algorithms.kmeans.KMeans;
import clustering.benchmark.config.AlgorithmSpec;
import clustering.benchmark.config.Params;
import clustering.core.Clusterer;
import clustering.core.Geometry;
import clustering.distance.DistanceMetric;
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
        m.put("fastpam", AlgorithmRegistry::fastpam);
        m.put("fasterpam", AlgorithmRegistry::fasterpam);
        m.put("distfastpam", AlgorithmRegistry::distfastpam);
        m.put("clara", AlgorithmRegistry::clara);
        m.put("pamae", AlgorithmRegistry::pamae);
        m.put("dbscanpp", AlgorithmRegistry::dbscanpp);
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

    /** Bisecting k-means — divisive, one FLIP-176 job for the whole split search (trials
     *  included). */
    private static Built bisectingkmeans(Map<String, Object> params) {
        Geometry geometry = geometryFrom(params);
        return new Built(new BisectingKMeans(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 20),
            Params.doubleParam(params, "eps", 1e-4),
            Params.longParam(params, "seed", 42L),
            geometry,
            // Steinbach et al.'s ITER: competing 2-means runs per split, cheapest wins.
            // 1 = the single-shot rung; sweeping it is the cost-vs-stability knob.
            Params.intParam(params, "trials", 1),
            // Which leaf gets split: 'cost' minimises error (scikit-learn's default),
            // 'size' balances the tree (what Steinbach et al. ran).
            Params.stringParam(params, "select", "cost")), geometry.modelDistance());
    }

    /** FastPAM1 — driver-local exact swap (Schubert &amp; Rousseeuw 2019), same result as the
     *  1990 PAM search. */
    private static Built fastpam(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new FastPAM(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 100),
            distance), distance);
    }

    /** FasterPAM — driver-local eager swap (Schubert &amp; Rousseeuw 2021). */
    private static Built fasterpam(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new FasterPAM(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 100),
            distance,
            Params.longParam(params, "seed", 42L)), distance);
    }

    /** Distributed FastPAM — O(n²) work split across machines, no driver-side matrix. */
    private static Built distfastpam(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new DistributedFastPAM(
            Params.intParam(params, "k"),
            Params.intParam(params, "maxIter", 100),
            distance), distance);
    }

    /** CLARA — the exact solver on random samples, scales to large datasets. One FLIP-176 job with
     *  the data cached once: round 0 draws every sample in a single pass and solves them, round 1
     *  scores every candidate against the full data in a single pass.
     *
     *  The separate `claraflip` entry (one sample per round) was folded into this one on 5.09.2026:
     *  batching inside the cached iteration does the same distance work in 2 passes instead of
     *  numSamples + 1, so it dominated both earlier entries and neither is worth keeping beside
     *  it. */
    private static Built clara(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new CLARA(
            Params.intParam(params, "k"),
            Params.intParam(params, "numSamples", 5),
            Params.intParam(params, "sampleSize", 1000),
            Params.intParam(params, "maxIter", 100),
            distance,
            // Which driver-local solver runs on each sample (Schubert & Rousseeuw 2021).
            // 'fastpam' is the exact rung — identical result to the 1990 PAM, O(k) cheaper.
            Params.stringParam(params, "inner", "fastpam"),
            Params.longParam(params, "seed", 42L)), distance);
    }

    /** PAMAE (KDD 2017) — parallel seeding (= CLARA) + parallel refinement over entire data. */
    private static Built pamae(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new PAMAE(
            Params.intParam(params, "k"),
            Params.intParam(params, "numSamples", 5),
            Params.intParam(params, "sampleSize", 1000),
            Params.intParam(params, "maxIter", 100),
            Params.intParam(params, "refineIters", 1),
            Params.intParam(params, "poolSize", 2000),
            distance,
            Params.stringParam(params, "inner", "fastpam"),
            Params.longParam(params, "seed", 42L)), distance);
    }

    /** DBSCAN++ — sampled candidate cores, exact densities against the full dataset. */
    private static Built dbscanpp(Map<String, Object> params) {
        DistanceMetric distance = distanceFrom(params);
        return new Built(new DBSCANpp(
            Params.doubleParam(params, "eps"),
            Params.intParam(params, "minPts"),
            // The universal accuracy-vs-cost knob; required, so no run hides which m it used.
            Params.doubleParam(params, "coreSampleFraction"),
            UniformSelection.fromName(
                Params.stringParam(params, "sampling", "uniform")),
            // assign: 'eps' = classic DBSCAN noise semantics, 'closest' = the paper's rule.
            assignWithinEps(Params.stringParam(params, "assign", "eps")),
            Params.intParam(params, "chunkSize", 2000),
            distance,
            Params.longParam(params, "seed", 42L)), distance);
    }

    public static Built create(AlgorithmSpec spec) {
        Function<Map<String, Object>, Built> f = REGISTRY.get(spec.name);
        return f.apply(spec.params == null ? new java.util.HashMap<>() : spec.params);
    }

    /** Parses the REQUIRED {@code distance} param. No default — every run must state its distance
     *  explicitly so benchmark results are unambiguous, and so a Flink config and a Spark config
     *  for the same experiment cannot silently disagree (Spark has always required it; defaulting
     *  here to euclidean meant a config that omitted it ran euclidean on one engine and failed on
     *  the other). Not used by the k-means family, whose metric is fixed by {@code geometry}. */
    private static DistanceMetric distanceFrom(Map<String, Object> params) {
        Object d = params.get("distance");
        if (d == null) {
            throw new IllegalArgumentException(
                "Missing required 'distance' parameter. Known: " + DistanceRegistry.knownNames());
        }
        return DistanceRegistry.get(d.toString());
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
