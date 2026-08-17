package clustering.algorithms.dbscan;

import clustering.TestFixtures;
import clustering.distance.EuclideanDistance;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The ε-graph phase has two implementations — driver-local and a distributed edge scan — and the
 *  whole point of having both is that they are interchangeable. So the property under test is
 *  equality of LABELS, not just of cluster counts: DBSCAN++ ids are part of the reproducibility
 *  measurements, so a run must not depend on which path its m happened to take.
 *
 *  Both are also checked against components computed from the definition (a plain BFS over the
 *  ε-graph), so "they agree" cannot mean "they are wrong in the same way". */
class EpsilonGraphComponentsSpec {

    private static final double Eps = 1.5;

    /** Three well-separated blobs, so there is more than one component and the labelling has
     *  something to get wrong. */
    private static double[][] cores(int count, long seed) {
        Random random = new Random(seed);
        double[][] centres = {{0.0, 0.0}, {20.0, 0.0}, {0.0, 20.0}};
        double[][] out = new double[count][];
        for (int i = 0; i < count; i++) {
            double[] centre = centres[i % centres.length];
            out[i] = new double[] {centre[0] + random.nextDouble(), centre[1] + random.nextDouble()};
        }
        return out;
    }

    /** Reference components straight from the definition: BFS over the ε-graph, labelled by
     *  ascending minimum member index (the numbering {@code UnionFind.componentIds} produces). */
    private static int[] componentsByDefinition(double[][] cores, double eps) {
        int[] labels = new int[cores.length];
        Arrays.fill(labels, -1);
        int next = 0;
        for (int start = 0; start < cores.length; start++) {
            if (labels[start] >= 0) {
                continue;
            }
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(start);
            labels[start] = next;
            while (!queue.isEmpty()) {
                int current = queue.poll();
                for (int other = 0; other < cores.length; other++) {
                    if (labels[other] < 0
                        && EuclideanDistance.INSTANCE.compute(cores[current], cores[other]) <= eps) {
                        labels[other] = next;
                        queue.add(other);
                    }
                }
            }
            next++;
        }
        return labels;
    }

    @Test
    void driverLocalPathMatchesComponentsFromTheDefinition() {
        double[][] points = cores(120, 7L);
        assertArrayEquals(componentsByDefinition(points, Eps),
            EpsilonGraphComponents.local(points, Eps, EuclideanDistance.INSTANCE));
    }

    @Test
    void distributedPathMatchesComponentsFromTheDefinition() {
        double[][] points = cores(120, 7L);
        assertArrayEquals(componentsByDefinition(points, Eps),
            EpsilonGraphComponents.distributed(points, Eps, EuclideanDistance.INSTANCE,
                TestFixtures.localEnvs(2), 2));
    }

    /** The distributed path streams edges back in whatever order subtasks produce them, so this is
     *  the check that the union-find's min-index rule really makes the labels order-free. 2 000
     *  rows across 4 subtasks interleaves the arrival order thoroughly. */
    @Test
    void bothPathsAgreeLabelForLabel() {
        double[][] points = cores(2000, 11L);
        int[] localLabels = EpsilonGraphComponents.local(points, Eps, EuclideanDistance.INSTANCE);
        int[] distributedLabels = EpsilonGraphComponents.distributed(points, Eps,
            EuclideanDistance.INSTANCE, TestFixtures.localEnvs(4), 4);

        assertArrayEquals(localLabels, distributedLabels);
        assertEquals(3, (int) Arrays.stream(localLabels).distinct().count(), "expected 3 blobs");
    }

    /** Parallelism must not change a single label — the same property, stated against the knob the
     *  scaling experiments sweep. */
    @Test
    void parallelismDoesNotChangeTheLabels() {
        double[][] points = cores(600, 5L);
        int[] reference = EpsilonGraphComponents.local(points, Eps, EuclideanDistance.INSTANCE);
        for (int parallelism : new int[] {1, 2, 3, 8}) {
            assertArrayEquals(reference,
                EpsilonGraphComponents.distributed(points, Eps, EuclideanDistance.INSTANCE,
                    TestFixtures.localEnvs(parallelism), parallelism),
                "labels changed at parallelism=" + parallelism);
        }
    }

    /** The balanced row mapping must be a BIJECTION on [0, m) — otherwise the scan would skip or
     *  duplicate rows, and a missing row means a silently missing edge. */
    @Test
    void balancedRowMappingIsABijection() {
        for (int m : new int[] {1, 2, 3, 17, 128}) {
            boolean[] seen = new boolean[m];
            for (long t = 0; t < m; t++) {
                int row = EpsilonGraphComponents.balancedRow(t, m);
                assertTrue(row >= 0 && row < m, "row out of range at m=" + m);
                assertTrue(!seen[row], "row " + row + " visited twice at m=" + m);
                seen[row] = true;
            }
        }
    }

    /** Pairing short rows with long ones is the reason contiguous index slices carry comparable
     *  work; without it the first subtask would scan ~m² pairs and the last almost none. */
    @Test
    void balancedRowMappingEvensOutTheWorkPerSlice() {
        int m = 10_000;
        int slices = 4;
        long[] pairsPerSlice = new long[slices];
        for (long t = 0; t < m; t++) {
            int row = EpsilonGraphComponents.balancedRow(t, m);
            pairsPerSlice[(int) (t * slices / m)] += m - row - 1L;
        }
        long min = Arrays.stream(pairsPerSlice).min().orElseThrow();
        long max = Arrays.stream(pairsPerSlice).max().orElseThrow();
        assertTrue(max <= min * 11 / 10, "slice work spread too wide: " + Arrays.toString(pairsPerSlice));
    }

    /** A short scan stays on the driver: the job's fixed cost would dwarf it. */
    @Test
    void aSmallCoreSetStaysOnTheDriver() {
        assertInstanceOf(EpsilonGraphComponents.Plan.DriverLocal.class,
            EpsilonGraphComponents.planFor(5000, 3, 192, 48, 1024L * 1024));
    }

    /** The measurement this guard exists for: on ONE machine the distributed path was 21.9 s
     *  against the driver-local 5.0 s (m = 100 000, 12 cores), because the local path already uses
     *  every core the driver has. A single-node run must therefore stay local NO MATTER how long
     *  the scan is — extra workers, not extra rows, are what make the job worth launching. */
    @Test
    void aSingleNodeRunStaysOnTheDriverEvenForALongScan() {
        EpsilonGraphComponents.Plan plan =
            EpsilonGraphComponents.planFor(1_000_000, 8, 12, 12, 64L * 1024 * 1024);
        assertTrue(((EpsilonGraphComponents.Plan.DriverLocal) plan).reason.contains("parallelism"),
            "the reason must name the cause: " + plan);
    }

    /** A long scan on a cluster with many more cores than the driver is distributed — and, unlike
     *  on Spark, DENSITY is not a reason to refuse: the edges are streamed into the union-find
     *  instead of collected per block, so there is no edge-list budget to blow. */
    @Test
    void aLongScanOnARealClusterIsDistributedRegardlessOfDensity() {
        assertInstanceOf(EpsilonGraphComponents.Plan.Distributed.class,
            EpsilonGraphComponents.planFor(500_000, 8, 192, 48, 32L * 1024 * 1024));
        // The Spark run that OOM-ed: 1.9 M cores, degree ~2·10⁴. Here only the shipped-coordinate
        // cap can stop it, and at 3 dims 1.9 M cores still fit.
        assertInstanceOf(EpsilonGraphComponents.Plan.Distributed.class,
            EpsilonGraphComponents.planFor(1_899_547, 3, 192, 48, 45L * 1024 * 1024));
    }

    /** What Flink DOES have to refuse: the core coordinates travel inside the JobGraph, so beyond
     *  a modest cap the phase stays on the driver rather than failing in job submission. */
    @Test
    void anUnshippableCoreSetStaysOnTheDriverAndSaysSo() {
        EpsilonGraphComponents.Plan plan =
            EpsilonGraphComponents.planFor(2_000_000, 512, 192, 48, 8L * 1024 * 1024 * 1024);
        assertTrue(((EpsilonGraphComponents.Plan.DriverLocal) plan).reason.contains("shipping"),
            "the reason must name the cause: " + plan);
    }

    @Test
    void singleCorePointIsItsOwnComponent() {
        double[][] one = {{1.0, 1.0}};
        assertArrayEquals(new int[] {0}, EpsilonGraphComponents.local(one, Eps, EuclideanDistance.INSTANCE));
        assertArrayEquals(new int[] {0}, EpsilonGraphComponents.distributed(one, Eps,
            EuclideanDistance.INSTANCE, TestFixtures.localEnvs(2), 2));
    }

    @Test
    void computePicksTheDriverPathForASmallCoreSetAndStillLabelsCorrectly() {
        double[][] points = cores(200, 3L);
        assertArrayEquals(componentsByDefinition(points, Eps),
            EpsilonGraphComponents.compute(points, Eps, EuclideanDistance.INSTANCE,
                TestFixtures.localEnvs(2), 2, 4.0));
    }
}
