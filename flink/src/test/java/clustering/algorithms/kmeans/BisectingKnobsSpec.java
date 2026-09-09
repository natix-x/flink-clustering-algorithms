package clustering.algorithms.kmeans;

import clustering.algorithms.kmeans.hierarchical.BisectingKMeans;
import clustering.algorithms.kmeans.hierarchical.BisectingKMeansModel;
import clustering.TestFixtures;
import clustering.core.EnvFactory;
import clustering.core.EuclideanGeometry;
import clustering.core.PointSource;
import clustering.core.Points;
import clustering.distance.EuclideanDistance;
import org.apache.flink.ml.linalg.DenseVector;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.TreeMap;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The two knobs on bisecting k-means, mirroring the Spark {@code BisectingLeafSelectionSpec} and
 *  {@code BisectingTrialsSpec}.
 *
 *  <ul>
 *    <li>{@code select} — {@code cost} (highest SSE, scikit-learn's default) vs {@code size}
 *        (most rows, what Steinbach, Karypis &amp; Kumar ran). They pursue different objectives, so
 *        the test's job is to show they really are different, not that one wins.</li>
 *    <li>{@code trials} — the paper's ITER. What it has to guarantee: {@code trials = 1} is the
 *        single-shot default; more trials never pick a costlier SPLIT (the winner is chosen by the
 *        same cost the tree minimises); and a fixed count is reproducible, so the variance a sweep
 *        reports is the platform's and not the knob's.</li>
 *  </ul>
 */
class BisectingKnobsSpec {

    /** The knob tests run at parallelism 1 ON PURPOSE. A knob is a property of the algorithm, and
     *  at parallelism 1 there is no rebalance choice to make — one subtask sees every point in
     *  source order — so the fit is deterministic and the test measures the knob and nothing else.
     *
     *  <p>At higher parallelism the same config does NOT reproduce: Flink's rebalance partitioner
     *  starts at a random channel, so the seed sample and therefore the tree vary between runs.
     *  That is left in place deliberately (see {@link BisectingKMeans}) and is characterised by
     *  {@link #theTreeVariesAcrossRunsAtHigherParallelismButStaysSane} below rather than asserted
     *  away. Demanding bit-identity here instead is what made two earlier versions of these tests
     *  fail for a reason that had nothing to do with the knobs. */
    private static final int PARALLELISM = 1;

    /** Where the platform variance is characterised instead of removed. */
    private static final int PARALLEL_RUNS_PARALLELISM = 4;

    /** Built so the two criteria MUST disagree: a populous, tight group (many rows, little error)
     *  against a sparse, scattered one (few rows, most of the error). After the first bisection
     *  separates them, {@code cost} reaches for the scattered leaf and {@code size} for the
     *  crowded one. */
    private static List<double[]> lopsided() {
        Random rng = new Random(3L);
        List<double[]> rows = new ArrayList<>();
        for (int i = 0; i < 400; i++) {
            rows.add(new double[] {rng.nextGaussian() * 0.3, rng.nextGaussian() * 0.3});
        }
        for (int i = 0; i < 50; i++) {
            rows.add(new double[] {60.0 + rng.nextGaussian() * 4.0, rng.nextGaussian() * 4.0});
        }
        return rows;
    }

    /** Six well-separated blobs: enough structure that a bad first bisection is recoverable by a
     *  later trial, so the trials actually disagree instead of all landing on the same split. */
    private static List<double[]> sixBlobs() {
        double[][] centres = {{0, 0}, {12, 0}, {0, 12}, {12, 12}, {24, 6}, {6, 24}};
        Random rng = new Random(7L);
        List<double[]> rows = new ArrayList<>();
        for (double[] c : centres) {
            for (int i = 0; i < 60; i++) {
                rows.add(new double[] {c[0] + rng.nextGaussian(), c[1] + rng.nextGaussian()});
            }
        }
        return rows;
    }

    private static BisectingKMeansModel fit(List<double[]> rows, int k, int trials, String select) {
        return fit(rows, k, trials, select, PARALLELISM);
    }

    private static BisectingKMeansModel fit(List<double[]> rows, int k, int trials, String select,
                                            int parallelism) {
        PointSource source = TestFixtures.source(rows);
        EnvFactory envs = TestFixtures.localEnvs(parallelism);
        return new BisectingKMeans(k, 20, 1e-4, 11L, EuclideanGeometry.INSTANCE, trials, select)
            .fit(source, envs, parallelism);
    }

    /** Leaf sizes, ascending — the tree's SHAPE, which is what `select` changes. */
    private static List<Long> leafSizes(List<double[]> rows, String select) {
        int[] labels = fit(rows, 3, 1, select).labels(Points.vectorsOf(rows));
        TreeMap<Integer, Long> byLabel = new TreeMap<>();
        for (int label : labels) {
            byLabel.merge(label, 1L, Long::sum);
        }
        List<Long> sizes = new ArrayList<>(byLabel.values());
        sizes.sort(null);
        return sizes;
    }

    /** Total SSE of the fitted leaves — the objective the tree greedily minimises, measured against
     *  the leaf a point is actually ROUTED to (root-to-leaf walk), not its globally nearest
     *  centroid. */
    private static double modelCost(List<double[]> rows, int k, int trials) {
        BisectingKMeansModel model = fit(rows, k, trials, "cost");
        DenseVector[] centroids = model.clusterCentroids();
        int[] labels = model.labels(Points.vectorsOf(rows));
        double total = 0.0;
        for (int i = 0; i < rows.size(); i++) {
            double d = EuclideanDistance.INSTANCE.compute(rows.get(i), centroids[labels[i]].values);
            total += d * d;
        }
        return total;
    }

    @Test
    void costAndSizePickDifferentLeavesSoTheyBuildDifferentTrees() {
        List<double[]> rows = lopsided();
        List<Long> byCost = leafSizes(rows, "cost");
        List<Long> bySize = leafSizes(rows, "size");

        // Both are valid 3-clusterings; only the shape differs.
        assertEquals(3, byCost.size());
        assertEquals(3, bySize.size());
        assertNotEquals(byCost, bySize, "expected different trees, got " + byCost + " for both");
        // `size` splits the 400-row group, so its largest leaf is smaller than `cost`'s, which
        // leaves that group whole and cuts the 50-row scattered one instead. This is the
        // balance-vs-error trade-off the knob exists to measure.
        assertTrue(bySize.get(2) < byCost.get(2),
            "size=" + bySize + " was not more balanced than cost=" + byCost);
    }

    @Test
    void costIsTheDefaultAndSelectIsCaseInsensitive() {
        List<double[]> rows = lopsided();
        assertEquals(leafSizes(rows, "cost"), leafSizes(rows, "COST"), "select is case-insensitive");

        PointSource source = TestFixtures.source(rows);
        EnvFactory envs = TestFixtures.localEnvs(PARALLELISM);
        int[] explicit = fit(rows, 3, 1, "cost").labels(Points.vectorsOf(rows));
        int[] byDefault = new BisectingKMeans(3, 20, 1e-4, 11L)
            .fit(source, envs, PARALLELISM).labels(Points.vectorsOf(rows));
        assertArrayEqualsLabels(explicit, byDefault);
    }

    /** k = 2 is the only k at which the tree IS one split, so the local guarantee is directly
     *  observable: the winner is chosen by exactly the cost measured here, therefore competing
     *  trials cannot lose to a single one. */
    @Test
    void moreTrialsNeverYieldACostlierSplit() {
        List<double[]> rows = sixBlobs();
        double one = modelCost(rows, 2, 1);
        double five = modelCost(rows, 2, 5);
        assertTrue(five <= one + 1e-9,
            "trials=5 split cost " + five + " exceeded trials=1 cost " + one);
    }

    /** The knob's guarantee is per-SPLIT, not per-tree: the tree is greedy, so a locally better
     *  first cut changes which leaf is picked in every later round and can land on a worse final
     *  objective. Asserted only as "the final cost may move in either direction" — pinning a
     *  direction would pin this data and seed, not the property. */
    @Test
    void moreTrialsDoNotGuaranteeABetterTree() {
        List<double[]> rows = sixBlobs();
        double one = modelCost(rows, 6, 1);
        double five = modelCost(rows, 6, 5);
        assertTrue(one > 0.0 && five > 0.0, "both trees must have a positive cost");
        // Deliberately no <=: greedy selection is not monotone in `trials`, and a sweep must be
        // read as cost-vs-stability rather than as a monotone improvement.
    }

    /** With the platform's partitioning choice removed, a fixed {@code trials} is reproducible —
     *  so the knob itself introduces no randomness of its own beyond the seed it is given. */
    @Test
    void aFixedTrialsCountIsReproducibleWhenPartitioningIsNotInPlay() {
        List<double[]> rows = sixBlobs();
        assertArrayEqualsLabels(
            fit(rows, 6, 4, "cost").labels(Points.vectorsOf(rows)),
            fit(rows, 6, 4, "cost").labels(Points.vectorsOf(rows)));
    }

    /** The variance the engine really has, characterised rather than removed: at parallelism > 1 the
     *  tree can differ between two runs of one config, because the seed sample is drawn from
     *  whatever rows a subtask happened to receive. What must NOT differ is the quality — a run that
     *  produced a wildly worse tree would be a bug, not variance.
     *
     *  Asserted as a bound, not an equality, which is the honest shape for a randomised method: k
     *  leaves every time, and every run's objective within a factor of the best. The spread itself
     *  is a number the benchmark reports through its repetitions. */
    @Test
    void theTreeVariesAcrossRunsAtHigherParallelismButStaysSane() {
        List<double[]> rows = sixBlobs();
        double best = Double.MAX_VALUE;
        double worst = 0.0;
        for (int run = 0; run < 4; run++) {
            BisectingKMeansModel model = fit(rows, 6, 1, "cost", PARALLEL_RUNS_PARALLELISM);
            assertEquals(6, model.numClusters(), "every run must still build k leaves");
            DenseVector[] centroids = model.clusterCentroids();
            int[] labels = model.labels(Points.vectorsOf(rows));
            double cost = 0.0;
            for (int i = 0; i < rows.size(); i++) {
                double d = EuclideanDistance.INSTANCE.compute(
                    rows.get(i), centroids[labels[i]].values);
                cost += d * d;
            }
            best = Math.min(best, cost);
            worst = Math.max(worst, cost);
        }
        assertTrue(worst <= best * 1.5,
            "run-to-run spread is variance, but " + worst + " against " + best
                + " is too wide to be that");
    }

    @Test
    void badKnobsAreRejectedBeforeAnyJobRuns() {
        assertTrue(assertThrows(IllegalArgumentException.class,
            () -> new BisectingKMeans(3, 20, 1e-4, 1L, EuclideanGeometry.INSTANCE, 1, "entropy"))
            .getMessage().contains("cost, size"));
        assertTrue(assertThrows(IllegalArgumentException.class,
            () -> new BisectingKMeans(3, 20, 1e-4, 1L, EuclideanGeometry.INSTANCE, 0, "cost"))
            .getMessage().contains("trials"));
    }

    private static void assertArrayEqualsLabels(int[] a, int[] b) {
        assertEquals(a.length, b.length);
        for (int i = 0; i < a.length; i++) {
            assertEquals(a[i], b[i], "label " + i);
        }
    }
}
