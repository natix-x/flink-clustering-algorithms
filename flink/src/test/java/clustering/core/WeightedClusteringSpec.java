package clustering.core;

import clustering.algorithms.dbscan.CoreLabelModel;
import clustering.algorithms.dbscan.DBSCANpp;
import clustering.algorithms.dbscan.components.UniformSelection;
import clustering.algorithms.kmeans.hierarchical.BisectingKMeans;
import clustering.algorithms.kmeans.hierarchical.BisectingKMeansModel;
import clustering.algorithms.kmedoids.distributed.DistributedFastPAM;
import clustering.algorithms.kmedoids.hybrid.CLARA;
import clustering.algorithms.kmedoids.hybrid.PAMAE;
import clustering.algorithms.kmedoids.local.FastPAM;
import clustering.TestFixtures;
import clustering.algorithms.kmeans.KMeans;
import clustering.algorithms.kmeans.KMeansModel;
import clustering.algorithms.kmedoids.KMedoidsModel;
import clustering.distance.EuclideanDistance;
import org.apache.flink.ml.linalg.DenseVector;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The load-bearing invariant of the weight support: <b>weighting equals duplication</b>. Clustering
 *  one row of weight w must equal clustering w copies of that row. Mirrors the Spark
 *  {@code WeightedClusteringSpec} / {@code WeightedMedoidsSpec}.
 *
 *  <p>Asserted on the OBJECTIVE rather than on prototype coordinates, and deliberately: duplicating
 *  a row makes each copy a separate candidate, so a medoid solver may legitimately fill two slots
 *  with two copies of one coordinate — a tied optimum, not a different one. The objective is what
 *  the algorithms actually minimise, so it is the thing that has to match.
 *
 *  <p>Every family is covered, because each one weights a different sum: k-means the mean and the
 *  SSE, bisecting additionally the leaf-selection mass, the medoid ladder the BUILD and swap deltas,
 *  and the sampling methods the full-data cost that picks their winner. A family that quietly
 *  dropped the weight would pass every other spec in the suite and fail only here. */
class WeightedClusteringSpec {

    private static final int PARALLELISM = 2;
    private static final double TOLERANCE = 1e-6;

    /** Three tight groups with DIFFERENT multiplicities, so an implementation that ignores weights
     *  lands somewhere else entirely rather than coincidentally agreeing. */
    private static final double[][] DISTINCT_ROWS = {
        {0.0, 0.0}, {0.4, 0.1}, {0.1, 0.4},
        {20.0, 0.0}, {20.3, 0.2},
        {0.0, 20.0}, {0.2, 20.4}, {0.3, 19.8}
    };
    private static final int[] MULTIPLICITIES = {7, 1, 3, 5, 2, 4, 6, 1};

    private static List<double[]> distinctRows() {
        List<double[]> rows = new ArrayList<>();
        for (double[] row : DISTINCT_ROWS) {
            rows.add(row.clone());
        }
        return rows;
    }

    private static double[] multiplicitiesAsWeights() {
        double[] weights = new double[MULTIPLICITIES.length];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = MULTIPLICITIES[i];
        }
        return weights;
    }

    /** The same data with each row repeated {@code MULTIPLICITIES[i]} times, all at weight 1. */
    private static List<double[]> duplicatedRows() {
        List<double[]> rows = new ArrayList<>();
        for (int i = 0; i < DISTINCT_ROWS.length; i++) {
            for (int copy = 0; copy < MULTIPLICITIES[i]; copy++) {
                rows.add(DISTINCT_ROWS[i].clone());
            }
        }
        return rows;
    }

    /** Σ w_x · min_p d(x, p) — the objective every entry here minimises, evaluated on the DUPLICATED
     *  data so both sides are scored against exactly the same population. */
    private static double cost(DenseVector[] prototypes) {
        double total = 0.0;
        for (double[] x : duplicatedRows()) {
            double min = Double.MAX_VALUE;
            for (DenseVector p : prototypes) {
                min = Math.min(min, EuclideanDistance.INSTANCE.compute(x, p.values));
            }
            total += min;
        }
        return total;
    }

    /** Fits {@code clusterer} twice — once on weighted rows, once on their duplication — and asserts
     *  the two land on the same objective. */
    private void assertWeightingEqualsDuplication(String what, Clusterer clusterer,
                                                  Function<Model, DenseVector[]> prototypesOf) {
        EnvFactory envs = TestFixtures.localEnvs(PARALLELISM);
        PointSource weighted =
            TestFixtures.weightedSource(distinctRows(), multiplicitiesAsWeights());
        PointSource duplicated = TestFixtures.source(duplicatedRows());

        double weightedCost =
            cost(prototypesOf.apply(clusterer.fit(weighted, envs, PARALLELISM)));
        double duplicatedCost =
            cost(prototypesOf.apply(clusterer.fit(duplicated, envs, PARALLELISM)));

        assertEquals(duplicatedCost, weightedCost, TOLERANCE,
            what + ": weighting must equal duplication — weighted=" + weightedCost
                + " duplicated=" + duplicatedCost);
    }

    private static DenseVector[] centroids(Model model) {
        return ((KMeansModel) model).centroids();
    }

    private static DenseVector[] medoids(Model model) {
        return ((KMedoidsModel) model).medoids();
    }

    /** k-means gets the invariant at {@code k = 1}, where it is exact and init-free: one centroid
     *  converges to the global WEIGHTED mean whatever the initialisation was, so the assertion tests
     *  the arithmetic and nothing else.
     *
     *  <p>The whole-fit comparison at {@code k > 1} is deliberately NOT asserted, because it is not
     *  a valid invariant: k-means init draws k rows, and duplication changes the row multiset it
     *  draws from, so the two runs legitimately start from different seeds and can converge to
     *  different local optima. Measured here: 5.82 against 220.91 on three tight groups — not a
     *  weighting bug, an init difference. No sampling-based init can be duplication-invariant, in
     *  either engine; Spark's equivalent assertion holds only because its draw happens to land on
     *  the same coordinates for that data. The families whose initialisation IS deterministic
     *  (bisecting's farthest-first bottom-k, the medoid ladder's BUILD) do get the full-fit
     *  assertion below, which is where it means something. */
    @Test
    void kMeansCentroidIsTheWeightedMean() {
        EnvFactory envs = TestFixtures.localEnvs(PARALLELISM);
        KMeans oneCluster = new KMeans(1, 30, 1e-12, 42L, EuclideanGeometry.INSTANCE);

        // A light point at 0 and a heavy one at 10: the weighted mean sits at 9, the unweighted
        // mean would sit at 5. So ignoring the weight is visible, not a rounding difference.
        List<double[]> twoRows = List.of(new double[] {0.0}, new double[] {10.0});
        DenseVector weighted = centroids(oneCluster.fit(
            TestFixtures.weightedSource(twoRows, new double[] {1.0, 9.0}), envs, PARALLELISM))[0];
        assertEquals(9.0, weighted.values[0], 1e-9,
            "expected the weighted mean 9.0, got " + weighted.values[0]);

        // And it must equal the mean of the duplication of the same input.
        List<double[]> duplicated = new ArrayList<>();
        duplicated.add(new double[] {0.0});
        for (int i = 0; i < 9; i++) {
            duplicated.add(new double[] {10.0});
        }
        DenseVector byDuplication =
            centroids(oneCluster.fit(TestFixtures.source(duplicated), envs, PARALLELISM))[0];
        assertEquals(byDuplication.values[0], weighted.values[0], 1e-9);
    }

    /** The weighted SSE is the other half of what k-means minimises, and it is also init-free at
     *  {@code k = 1}: with the centroid pinned to the weighted mean, the reported objective must be
     *  the one the duplicated data produces. */
    @Test
    void kMeansErrorIsWeighted() {
        EnvFactory envs = TestFixtures.localEnvs(PARALLELISM);
        KMeans oneCluster = new KMeans(1, 30, 1e-12, 42L, EuclideanGeometry.INSTANCE);

        List<double[]> rows = distinctRows();
        DenseVector[] weighted = centroids(oneCluster.fit(
            TestFixtures.weightedSource(rows, multiplicitiesAsWeights()), envs, PARALLELISM));
        DenseVector[] duplicated =
            centroids(oneCluster.fit(TestFixtures.source(duplicatedRows()), envs, PARALLELISM));

        assertEquals(cost(duplicated), cost(weighted), TOLERANCE,
            "the single-cluster objective must not depend on how the mass was expressed");
    }

    @Test
    void bisectingKMeansWeightsEveryLeafStatistic() {
        assertWeightingEqualsDuplication("bisectingkmeans",
            new BisectingKMeans(3, 30, 1e-9, 42L, EuclideanGeometry.INSTANCE, 1, "cost"),
            model -> ((BisectingKMeansModel) model).clusterCentroids());
    }

    /** {@code select: size} is the one criterion that reads mass DIRECTLY, so it gets its own case:
     *  a row-counting version would call a 1-row leaf of weight 7 small. */
    @Test
    void bisectingSizeSelectionReadsMassNotRows() {
        assertWeightingEqualsDuplication("bisectingkmeans select=size",
            new BisectingKMeans(3, 30, 1e-9, 42L, EuclideanGeometry.INSTANCE, 1, "size"),
            model -> ((BisectingKMeansModel) model).clusterCentroids());
    }

    @Test
    void fastPamWeightsBuildAndSwap() {
        assertWeightingEqualsDuplication("fastpam",
            new FastPAM(3, 50, EuclideanDistance.INSTANCE),
            WeightedClusteringSpec::medoids);
    }

    @Test
    void distributedFastPamWeightsItsDeltas() {
        assertWeightingEqualsDuplication("distfastpam",
            new DistributedFastPAM(3, 50, EuclideanDistance.INSTANCE),
            WeightedClusteringSpec::medoids);
    }

    /** The sampling methods are the interesting case: the weight has to survive the DRAW, not just
     *  the arithmetic — the sample carries its rows' weights into the local solver, and the
     *  full-data cost that picks the winner is weighted too.
     *
     *  Sampling is uniform over ROWS, so a weighted run and its duplication do not draw the same
     *  rows: the assertion is that both reach the same optimum, not that they drew the same sample.
     *
     *  <p>Hence {@code numSamples = 16} rather than the default handful, and it is load-bearing.
     *  The two sides are NOT symmetric here: the weighted input is 8 distinct rows, so with
     *  {@code sampleSize = 8} it takes CLARA's whole-dataset branch and returns the exact optimum,
     *  while its 29-row duplication really samples. Covering the three groups is not enough for the
     *  duplication to match that optimum — group C's medoid depends on WHICH of its rows were
     *  drawn ({0.2, 20.4} beats {0.0, 20.0} by 0.647 on the full data) — so with only a few samples
     *  the assertion measured draw luck and failed about one run in three, at any seed: Flink's
     *  rebalance partitioner starts at a random channel, so a fixed seed does not fix which rows a
     *  subtask holds. Raising the number of INDEPENDENT samples is CLARA's own mechanism for
     *  reliability and the honest lever here; 16 gave 0 failures in 12 trials across 6 seeds, where
     *  4 gave 2 in 6. */
    @Test
    void claraCarriesWeightsThroughTheSampleAndTheCostJob() {
        assertWeightingEqualsDuplication("clara",
            new CLARA(3, 16, 8, 50, EuclideanDistance.INSTANCE, "fastpam", 42L),
            WeightedClusteringSpec::medoids);
    }

    /** {@code poolSize = 40} is the load-bearing knob here, for the same asymmetry the CLARA test
     *  above documents: the weighted input is 8 distinct rows and its duplication is 29, so only
     *  the duplication really samples. Phase II can repair a seeding that missed group C's best
     *  representative ONLY if that row is in its candidate pool — at the old {@code poolSize = 8}
     *  the pool was 8 of 29 rows, it frequently was not, and the assertion failed about one run in
     *  six. A pool that covers the whole duplication makes the test say what it means to say: that
     *  weighted REFINEMENT reaches the same objective as refinement over the duplication, rather
     *  than that the seeding got lucky. Verified: 0 failures in 6 seeds at poolSize 40, 1 in 6 at
     *  poolSize 8. */
    @Test
    void pamaeWeightsSeedingAndRefinement() {
        assertWeightingEqualsDuplication("pamae",
            new PAMAE(3, 4, 8, 50, 3, 40, EuclideanDistance.INSTANCE, "fastpam", 42L),
            WeightedClusteringSpec::medoids);
    }

    /** DBSCAN++'s weight semantics are its own: {@code minPts} becomes a threshold on MASS, not on
     *  rows, because a point of weight w stands for w neighbours. That is what makes weighting equal
     *  duplication for the CORE-POINT test as well as for the objective — and it is the only place
     *  in the repo where a weight changes a discrete yes/no decision rather than a sum. */
    @Test
    void dbscanPlusPlusTreatsMinPtsAsAMassThreshold() {
        EnvFactory envs = TestFixtures.localEnvs(PARALLELISM);
        // Three pairs, each pair within eps of itself and far from the others. Unweighted every pair
        // has mass 2; weighted, pair i has mass 2·w_i. minPts = 8 therefore keeps only the pairs
        // whose weight is at least 4.
        List<double[]> rows = List.of(
            new double[] {0.0, 0.0}, new double[] {0.4, 0.0},
            new double[] {10.0, 0.0}, new double[] {10.4, 0.0},
            new double[] {30.0, 0.0}, new double[] {30.4, 0.0});
        double[] weights = {1.0, 1.0, 5.0, 5.0, 4.0, 4.0};

        CoreLabelModel model =
            (CoreLabelModel) new DBSCANpp(
                1.0, 8, 1.0,
                UniformSelection.INSTANCE,
                true, 100, EuclideanDistance.INSTANCE, 42L)
                .fit(TestFixtures.weightedSource(rows, weights), envs, PARALLELISM);

        List<Double> coreXs = new ArrayList<>();
        for (double[] core : model.corePoints()) {
            coreXs.add(core[0]);
        }
        coreXs.sort(null);
        assertEquals(List.of(10.0, 10.4, 30.0, 30.4), coreXs,
            "expected only the weight-5 and weight-4 pairs to reach a mass of 8, got " + coreXs);
    }

    /** A run with no weights must be bit-identical to the unit-weight run, or every number already
     *  measured on this engine changes meaning. */
    @Test
    void unitWeightsAreTheUnweightedRun() {
        List<double[]> rows = duplicatedRows();
        EnvFactory envs = TestFixtures.localEnvs(PARALLELISM);
        KMeans kmeans = new KMeans(3, 30, 1e-9, 42L, EuclideanGeometry.INSTANCE);

        DenseVector[] implicitUnit =
            centroids(kmeans.fit(TestFixtures.source(rows), envs, PARALLELISM));
        DenseVector[] explicitUnit = centroids(kmeans.fit(
            TestFixtures.weightedSource(rows, Weights.unit(rows.size())), envs, PARALLELISM));

        assertEquals(implicitUnit.length, explicitUnit.length);
        for (int i = 0; i < implicitUnit.length; i++) {
            assertArrayEquals(implicitUnit[i].values, explicitUnit[i].values);
        }
    }

    /** The record's serializer is what carries the weight across the network and into the FLIP-176
     *  caches, so a round-trip through it is worth pinning directly — a dropped weight there would
     *  look like an algorithm bug everywhere else. */
    @Test
    void theRecordSurvivesASerializerRoundTrip() throws Exception {
        Random rng = new Random(5L);
        for (int dimension : new int[] {1, 3, 8, 256}) {
            double[] coords = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                coords[i] = rng.nextGaussian();
            }
            WeightedPoint original = new WeightedPoint(new DenseVector(coords), 3.25);

            org.apache.flink.core.memory.DataOutputSerializer out =
                new org.apache.flink.core.memory.DataOutputSerializer(64);
            clustering.core.WeightedPointSerializer.INSTANCE.serialize(original, out);
            WeightedPoint restored = clustering.core.WeightedPointSerializer.INSTANCE.deserialize(
                new org.apache.flink.core.memory.DataInputDeserializer(out.getCopyOfBuffer()));

            assertArrayEquals(coords, restored.values());
            assertEquals(3.25, restored.weight, 0.0);
        }
    }

    /** The reuse path is the one the caches actually take, and it must not leak the previous
     *  record's state — a stale weight or a stale coordinate would be invisible until a fold
     *  produced a subtly wrong sum. */
    @Test
    void theReuseDeserialisationPathDoesNotLeakState() throws Exception {
        WeightedPoint reuse = new WeightedPoint();
        double[][] records = {{1.0, 2.0}, {7.0, 8.0, 9.0}, {4.0}};
        double[] weights = {1.0, 5.5, 0.25};

        for (int i = 0; i < records.length; i++) {
            org.apache.flink.core.memory.DataOutputSerializer out =
                new org.apache.flink.core.memory.DataOutputSerializer(32);
            clustering.core.WeightedPointSerializer.INSTANCE.serialize(
                new WeightedPoint(new DenseVector(records[i]), weights[i]), out);
            WeightedPoint got = clustering.core.WeightedPointSerializer.INSTANCE.deserialize(
                reuse, new org.apache.flink.core.memory.DataInputDeserializer(out.getCopyOfBuffer()));

            assertArrayEquals(records[i], got.values(), "record " + i);
            assertEquals(weights[i], got.weight, 0.0, "weight " + i);
        }
    }

    private static void assertArrayEquals(double[] expected, double[] actual) {
        assertArrayEquals(expected, actual, "coordinates");
    }

    private static void assertArrayEquals(double[] expected, double[] actual, String what) {
        assertEquals(expected.length, actual.length, what + " length");
        for (int i = 0; i < expected.length; i++) {
            assertTrue(expected[i] == actual[i],
                what + "[" + i + "]: expected " + Arrays.toString(expected)
                    + " got " + Arrays.toString(actual));
        }
    }
}
