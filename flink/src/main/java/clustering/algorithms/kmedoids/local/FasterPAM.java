package clustering.algorithms.kmedoids.local;

import clustering.algorithms.kmedoids.KMedoidsModel;
import clustering.algorithms.kmedoids.components.DistanceMatrix;
import clustering.algorithms.kmedoids.components.DriverSample;
import clustering.algorithms.kmedoids.components.MedoidBuildPhase;
import clustering.algorithms.kmedoids.components.NearestMedoidCache;
import clustering.algorithms.kmedoids.components.SwapDeltas;
import clustering.algorithms.kmedoids.components.SwapMove;
import clustering.distance.DistanceMetric;

/**
 * FasterPAM implementation (Schubert & Rousseeuw, 2021).
 * Performs eager swaps where improving candidates are immediately applied and caches are refreshed.
 */
public class FasterPAM implements DriverLocalKMedoids {

    private final int targetK;
    private final int maxIterations;
    private final DistanceMetric distanceMetric;
    private final long seed;

    public FasterPAM(int targetK, int maxIterations, DistanceMetric distanceMetric, long seed) {
        this.targetK = targetK;
        this.maxIterations = maxIterations;
        this.distanceMetric = distanceMetric;
        this.seed = seed;
    }

    @Override
    public KMedoidsModel fitLocal(double[][] points, double[] weights) {
        int numPoints = points.length;

        DistanceMatrix distanceMatrix = DistanceMatrix.computePairwise(points, distanceMetric);
        int[] medoids = MedoidBuildPhase.selectInitialMedoids(distanceMatrix, targetK, weights);
        NearestMedoidCache medoidCache = new NearestMedoidCache(numPoints);
        medoidCache.recompute(distanceMatrix, medoids);

        boolean[] isMedoid = new boolean[numPoints];
        for (int medoid : medoids) {
            isMedoid[medoid] = true;
        }

        int[] visitOrder = DriverSample.generateRandomPermutation(numPoints, seed);
        double[] slotDeltas = new double[targetK];

        int pass = 0;
        boolean hasImproved = true;

        while (hasImproved && pass < maxIterations) {
            hasImproved = false;
            for (int i = 0; i < numPoints; i++) {
                int candidate = visitOrder[i];
                if (isMedoid[candidate]) {
                    continue;
                }
                SwapMove bestMove = SwapDeltas.findBestMove(distanceMatrix, candidate, weights, medoidCache, slotDeltas);

                if (SwapMove.isImprovement(bestMove)) {
                    isMedoid[medoids[bestMove.targetSlot]] = false;
                    medoids[bestMove.targetSlot] = candidate;
                    isMedoid[candidate] = true;
                    // Incremental update, not a full O(n·k) recompute: only slot targetSlot moved,
                    // so almost every point's top-2 is untouched. See NearestMedoidCache.
                    medoidCache.updateAfterSwap(distanceMatrix, medoids, bestMove.targetSlot);
                    hasImproved = true;
                }
            }
            pass++;
        }

        return DriverLocalKMedoids.modelOf(points, medoids, distanceMetric);
    }
}
