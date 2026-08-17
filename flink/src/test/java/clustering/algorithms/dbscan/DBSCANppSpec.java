package clustering.algorithms.dbscan;

import clustering.core.Points;
import clustering.TestFixtures;
import clustering.benchmark.config.AlgorithmSpec;
import clustering.benchmark.registry.AlgorithmRegistry;
import clustering.core.EnvFactory;
import clustering.core.PointSource;
import clustering.distance.EuclideanDistance;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Validates {@code dbscanpp} against a textbook DBSCAN written from the definition — the same
 *  check the Spark {@code DBSCANppSpec} makes, so both engines are pinned to the definition
 *  rather than to each other.
 *
 *  At {@code coreSampleFraction = 1.0} DBSCAN++ IS classic DBSCAN, so the distributed
 *  implementation must produce the same partition as a straight reading of the definition. The
 *  reference below is deliberately dumb — plain Java, O(n²), no Flink — so it can be read against
 *  the definition line by line. */
class DBSCANppSpec {

    private static final double Eps = 0.5;
    private static final int MinPts = 4;

    /** Two 5x5 grids of spacing 0.3 (each a single eps-connected dense blob) plus one far
     *  outlier. Deterministic by construction — no RNG, so the comparison cannot flake. */
    private static List<double[]> fixture() {
        List<double[]> points = new ArrayList<>();
        points.addAll(TestFixtures.grid(0.0, 0.0, 5, 0.3));
        points.addAll(TestFixtures.grid(10.0, 10.0, 5, 0.3));
        points.add(new double[] {50.0, 50.0});
        return points;
    }

    private static final int OutlierIndex = 50;

    /** Classic DBSCAN, straight from the definition.
     *
     *  Core point: at least {@code minPts} points within eps, itself included. Two core points
     *  are in the same cluster when a chain of eps-steps between core points joins them. Every
     *  other point joins a cluster if some core point is within eps of it, else it is noise. */
    private static int[] referenceDBSCAN(List<double[]> points, double eps, int minPts) {
        int n = points.size();
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                d[i][j] = EuclideanDistance.INSTANCE.compute(points.get(i), points.get(j));
            }
        }
        boolean[] core = new boolean[n];
        for (int i = 0; i < n; i++) {
            int within = 0;
            for (int j = 0; j < n; j++) {
                if (d[i][j] <= eps) {
                    within++;
                }
            }
            core[i] = within >= minPts;
        }
        // Components of the eps-graph over core points, by repeated label relaxation — slow and
        // obviously correct, which is the point of a reference implementation.
        int[] comp = new int[n];
        for (int i = 0; i < n; i++) {
            comp[i] = i;
        }
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i < n; i++) {
                for (int j = 0; core[i] && j < n; j++) {
                    if (core[j] && d[i][j] <= eps) {
                        int lo = Math.min(comp[i], comp[j]);
                        if (comp[i] != lo || comp[j] != lo) {
                            comp[i] = lo;
                            comp[j] = lo;
                            changed = true;
                        }
                    }
                }
            }
        }
        int[] labels = new int[n];
        Arrays.fill(labels, -1);
        for (int i = 0; i < n; i++) {
            if (core[i]) {
                labels[i] = comp[i];
            } else {
                int nearestCore = -1;
                for (int j = 0; j < n; j++) {
                    if (core[j] && (nearestCore < 0 || d[i][j] < d[i][nearestCore])) {
                        nearestCore = j;
                    }
                }
                if (nearestCore >= 0 && d[i][nearestCore] <= eps) {
                    labels[i] = comp[nearestCore];
                }
            }
        }
        return labels;
    }

    private static DBSCANpp exact() {
        return new DBSCANpp(Eps, MinPts, 1.0, UniformSelection.INSTANCE, true, 2000,
            EuclideanDistance.INSTANCE, 42L);
    }

    @Test
    void coreSampleFractionOneReproducesTextbookDbscan() {
        List<double[]> points = fixture();
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        int[] reference = referenceDBSCAN(points, Eps, MinPts);
        assertEquals(2, TestFixtures.clustersOf(reference).size(),
            "fixture broken: the reference must find two clusters");

        int[] labels = exact().fit(source, envs, 2).labels(Points.wrapAll(points));

        assertEquals(TestFixtures.clustersOf(reference), TestFixtures.clustersOf(labels),
            "exact DBSCAN++ must reproduce the textbook partition");
        assertEquals(TestFixtures.noiseOf(reference), TestFixtures.noiseOf(labels));
        assertEquals(Set.of(OutlierIndex), TestFixtures.noiseOf(labels),
            "the far outlier must be the only noise point");
    }

    @Test
    void exactRunFindsExactlyTheTwoBlobs() {
        CoreLabelModel model = exact().fit(TestFixtures.source(fixture()), TestFixtures.localEnvs(2), 2);
        assertEquals(2, model.numClusters());
    }

    /** The invariant sampling must preserve at every s: a cluster never spans two blobs.
     *
     *  The converse is NOT asserted, on purpose. Sub-sampling core candidates thins the
     *  eps-graph, so gaps wider than eps can open inside one true blob and SPLIT it — that is
     *  DBSCAN++ behaving as published (the guarantee is asymptotic in m), and it is exactly the
     *  accuracy loss the s-sweep is supposed to measure. */
    @Test
    void sampledRunsNeverMergeTheTwoBlobs() {
        List<double[]> points = fixture();
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        for (double s : new double[] {0.5, 0.8}) {
            CoreLabelModel model = new DBSCANpp(Eps, MinPts, s, UniformSelection.INSTANCE, true,
                2000, EuclideanDistance.INSTANCE, 7L).fit(source, envs, 2);
            int[] labels = model.labels(Points.wrapAll(points));
            Set<Set<Integer>> clusters = TestFixtures.clustersOf(labels);

            assertTrue(!clusters.isEmpty(), "s=" + s + " produced no cluster at all");
            for (Set<Integer> cluster : clusters) {
                long blobs = cluster.stream().map(i -> points.get(i)[0] < 5.0).distinct().count();
                assertEquals(1L, blobs, "s=" + s + " produced a cluster spanning both blobs");
            }
            assertTrue(TestFixtures.noiseOf(labels).contains(OutlierIndex),
                "s=" + s + " lost the far outlier from the noise set");
        }
    }

    @Test
    void assignClosestReproducesThePaperAndEmitsNoNoise() {
        List<double[]> points = fixture();
        CoreLabelModel model = new DBSCANpp(Eps, MinPts, 1.0, UniformSelection.INSTANCE, false,
            2000, EuclideanDistance.INSTANCE, 42L)
            .fit(TestFixtures.source(points), TestFixtures.localEnvs(2), 2);

        assertTrue(TestFixtures.noiseOf(model.labels(Points.wrapAll(points))).isEmpty(),
            "without the eps condition every point must get a cluster");
    }

    @Test
    void allThreeSamplingStrategiesProduceAUsableClustering() {
        List<double[]> points = fixture();
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        for (String name : new String[] {"uniform", "linspace", "kcenter"}) {
            CoreLabelModel model = new DBSCANpp(Eps, MinPts, 0.6,
                CandidateSelectionStrategy.fromName(name, 3), true, 2000,
                EuclideanDistance.INSTANCE, 3L).fit(source, envs, 2);
            assertTrue(model.numClusters() >= 1, "sampling '" + name + "' produced no cluster");
            assertTrue(model.corePoints().length > 0, "sampling '" + name + "' produced no core point");
        }
    }

    @Test
    void parametersWithoutAnyCorePointYieldAnAllNoiseModel() {
        List<double[]> points = fixture();
        CoreLabelModel model = new DBSCANpp(0.01, 10, 1.0, UniformSelection.INSTANCE, true, 2000,
            EuclideanDistance.INSTANCE, 42L)
            .fit(TestFixtures.source(points), TestFixtures.localEnvs(2), 2);

        assertEquals(0, model.numClusters());
        int[] labels = model.labels(Points.wrapAll(points));
        assertTrue(TestFixtures.clustersOf(labels).isEmpty());
        assertEquals(points.size(), TestFixtures.noiseOf(labels).size());
    }

    /** Chunking must not change a single count: the chunk loop is a memory bound, not an
     *  approximation. */
    @Test
    void chunkSizeDoesNotChangeTheResult() {
        List<double[]> points = fixture();
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        int[] oneChunk = new DBSCANpp(Eps, MinPts, 1.0, UniformSelection.INSTANCE, true, 2000,
            EuclideanDistance.INSTANCE, 42L).fit(source, envs, 2).labels(Points.wrapAll(points));
        int[] manyChunks = new DBSCANpp(Eps, MinPts, 1.0, UniformSelection.INSTANCE, true, 7,
            EuclideanDistance.INSTANCE, 42L).fit(source, envs, 2).labels(Points.wrapAll(points));

        assertArrayEqualsInts(oneChunk, manyChunks);
    }

    @Test
    void dbscanExactAliasIsTheSCodePathWithSPinnedToOne() {
        List<double[]> points = fixture();
        PointSource source = TestFixtures.source(points);
        EnvFactory envs = TestFixtures.localEnvs(2);

        Map<String, Object> params = new HashMap<>();
        params.put("eps", Eps);
        params.put("minPts", MinPts);
        params.put("distance", "euclidean");
        AlgorithmSpec spec = new AlgorithmSpec();
        spec.name = "dbscanexact";
        spec.params = params;

        int[] viaAlias = AlgorithmRegistry.create(spec).clusterer.fit(source, envs, 2).labels(Points.wrapAll(points));
        int[] direct = exact().fit(source, envs, 2).labels(Points.wrapAll(points));
        assertArrayEqualsInts(viaAlias, direct);
    }

    @Test
    void dbscanExactRejectsSampledParamsInsteadOfIgnoringThem() {
        Map<String, Object> sampled = new HashMap<>();
        sampled.put("eps", Eps);
        sampled.put("minPts", MinPts);
        sampled.put("coreSampleFraction", 0.3);
        AlgorithmSpec spec = new AlgorithmSpec();
        spec.name = "dbscanexact";
        spec.params = sampled;
        assertTrue(assertThrows(IllegalArgumentException.class, () -> AlgorithmRegistry.create(spec))
            .getMessage().contains("exact by definition"));

        for (String key : new String[] {"sampling", "poolFactor"}) {
            Map<String, Object> params = new HashMap<>();
            params.put("eps", Eps);
            params.put("minPts", MinPts);
            params.put(key, key.equals("sampling") ? "kcenter" : 3);
            AlgorithmSpec withSampling = new AlgorithmSpec();
            withSampling.name = "dbscanexact";
            withSampling.params = params;
            assertTrue(assertThrows(IllegalArgumentException.class,
                () -> AlgorithmRegistry.create(withSampling)).getMessage().contains(key));
        }
    }

    @Test
    void clusterIdsAreReproducibleAcrossRepeatedFits() {
        PointSource source = TestFixtures.source(fixture());
        EnvFactory envs = TestFixtures.localEnvs(2);
        assertArrayEqualsInts(
            exact().fit(source, envs, 2).coreClusterLabels(),
            exact().fit(source, envs, 2).coreClusterLabels());
    }

    private static void assertArrayEqualsInts(int[] a, int[] b) {
        assertEquals(a.length, b.length);
        for (int i = 0; i < a.length; i++) {
            assertEquals(a[i], b[i], "entry " + i);
        }
    }
}
