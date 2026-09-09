package clustering.algorithms.dbscan;

import clustering.algorithms.dbscan.components.EpsilonGraphComponents;
import clustering.TestFixtures;
import clustering.distance.EuclideanDistance;
import org.junit.jupiter.api.Test;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The ε-graph phase runs as one distributed Flink job (no driver-local fallback — always worth
 *  its fixed cost per the 5.09.2026 decision to always pay for the distributed path). Checked
 *  against components computed from the definition (a plain BFS over the ε-graph), so "correct"
 *  cannot mean "wrong in a way the test shares". */
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
    void distributedLabelsMatchAcrossParallelism() {
        double[][] points = cores(2000, 11L);
        int[] reference = EpsilonGraphComponents.distributed(points, Eps, EuclideanDistance.INSTANCE,
            TestFixtures.localEnvs(1), 1);

        for (int parallelism : new int[] {1, 2, 3, 4, 8}) {
            assertArrayEquals(reference,
                EpsilonGraphComponents.distributed(points, Eps, EuclideanDistance.INSTANCE,
                    TestFixtures.localEnvs(parallelism), parallelism),
                "labels changed at parallelism=" + parallelism);
        }
        assertArrayEquals(componentsByDefinition(points, Eps), reference);
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

    @Test
    void singleCorePointIsItsOwnComponent() {
        double[][] one = {{1.0, 1.0}};
        assertArrayEquals(new int[] {0}, EpsilonGraphComponents.distributed(one, Eps,
            EuclideanDistance.INSTANCE, TestFixtures.localEnvs(2), 2));
    }

    @Test
    void computeMatchesComponentsFromTheDefinition() {
        double[][] points = cores(200, 3L);
        assertArrayEquals(componentsByDefinition(points, Eps),
            EpsilonGraphComponents.compute(points, Eps, EuclideanDistance.INSTANCE,
                TestFixtures.localEnvs(2), 2, 4.0));
    }
}
