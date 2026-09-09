package clustering.algorithms.kmedoids;

import clustering.algorithms.kmedoids.components.DistanceMatrix;
import clustering.algorithms.kmedoids.components.NearestMedoidCache;
import clustering.distance.EuclideanDistance;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** {@link NearestMedoidCache#updateAfterSwap} must produce EXACTLY the state a full
 *  {@link NearestMedoidCache#recompute} would, for every point, every time — it is a performance
 *  change (O(n) + a rescan of the affected points, instead of O(n·k) always), never an approximate
 *  one. Random swaps are the adversary: a hand-picked example can dodge every branch of the
 *  three-way split the update does, a thousand random ones over several k cannot.
 */
class NearestMedoidCacheSpec {

    private static double[][] randomPoints(int n, int dims, long seed) {
        Random rng = new Random(seed);
        double[][] points = new double[n][dims];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dims; d++) {
                points[i][d] = rng.nextGaussian();
            }
        }
        return points;
    }

    /** Asserts the two caches agree on all four arrays, point by point. */
    private static void assertSameState(NearestMedoidCache expected, NearestMedoidCache actual, String where) {
        assertArrayEquals(expected.closestDistances, actual.closestDistances, 1e-9, where + ": closestDistances");
        assertArrayEquals(expected.secondClosestDistances, actual.secondClosestDistances, 1e-9, where + ": secondClosestDistances");
        assertArrayEquals(expected.closestMedoidSlots, actual.closestMedoidSlots, where + ": closestMedoidSlots");
        assertArrayEquals(expected.secondMedoidSlots, actual.secondMedoidSlots, where + ": secondMedoidSlots");
    }

    @Test
    void updateAfterSwapMatchesFullRecomputeAcrossManyRandomSwaps() {
        int n = 300;
        int dims = 4;
        double[][] points = randomPoints(n, dims, 1L);
        DistanceMatrix distances = DistanceMatrix.computePairwise(points, EuclideanDistance.INSTANCE);

        for (int k : new int[] {1, 2, 5, 20}) {
            Random rng = new Random(100L + k);
            int[] medoids = new int[k];
            boolean[] isMedoid = new boolean[n];
            for (int slot = 0; slot < k; slot++) {
                int candidate;
                do {
                    candidate = rng.nextInt(n);
                } while (isMedoid[candidate]);
                medoids[slot] = candidate;
                isMedoid[candidate] = true;
            }

            NearestMedoidCache incremental = new NearestMedoidCache(n);
            incremental.recompute(distances, medoids);

            for (int swap = 0; swap < 500; swap++) {
                int slot = rng.nextInt(k);
                int newMedoid;
                do {
                    newMedoid = rng.nextInt(n);
                } while (isMedoid[newMedoid]);

                isMedoid[medoids[slot]] = false;
                medoids[slot] = newMedoid;
                isMedoid[newMedoid] = true;

                incremental.updateAfterSwap(distances, medoids, slot);

                NearestMedoidCache fromScratch = new NearestMedoidCache(n);
                fromScratch.recompute(distances, medoids);

                assertSameState(fromScratch, incremental, "k=" + k + " swap#" + swap);
            }
        }
    }

    /** k=1 is the edge case where "second nearest" never exists — pinned on its own since the
     *  three-way split leans on {@code secondMedoidSlots[i] == -1} behaving like "never matches". */
    @Test
    void updateAfterSwapHandlesSingleMedoidCorrectly() {
        double[][] points = randomPoints(50, 3, 7L);
        DistanceMatrix distances = DistanceMatrix.computePairwise(points, EuclideanDistance.INSTANCE);
        int[] medoids = {0};

        NearestMedoidCache cache = new NearestMedoidCache(50);
        cache.recompute(distances, medoids);

        medoids[0] = 17;
        cache.updateAfterSwap(distances, medoids, 0);

        NearestMedoidCache expected = new NearestMedoidCache(50);
        expected.recompute(distances, medoids);

        assertSameState(expected, cache, "k=1");
    }
}
