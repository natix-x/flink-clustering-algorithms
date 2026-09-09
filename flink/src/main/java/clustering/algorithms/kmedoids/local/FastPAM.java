package clustering.algorithms.kmedoids.local;

import clustering.algorithms.kmedoids.KMedoidsModel;
import clustering.algorithms.kmedoids.components.DistanceMatrix;
import clustering.algorithms.kmedoids.components.MedoidBuildPhase;
import clustering.algorithms.kmedoids.components.NearestMedoidCache;
import clustering.algorithms.kmedoids.components.SwapDeltas;
import clustering.algorithms.kmedoids.components.SwapMove;
import clustering.algorithms.kmedoids.distributed.DistributedFastPAM;
import clustering.distance.DistanceMetric;

import java.util.Arrays;

/**
 * FastPAM1 implementation (Schubert & Rousseeuw 2019).
 * Performs the exact PAM search by evaluating all (slot, candidate) pairs,
 * computing the cost delta for all k slots in a single O(n) pass.
 */
public class FastPAM implements DriverLocalKMedoids {

    private final int targetK;
    private final int maxIterations;
    private final DistanceMetric distanceMetric;

    public FastPAM(int targetK, int maxIterations, DistanceMetric distanceMetric) {
        this.targetK = targetK;
        this.maxIterations = maxIterations;
        this.distanceMetric = distanceMetric;
    }

    @Override
    public KMedoidsModel fitLocal(double[][] points, double[] weights) {
        int numPoints = points.length;

        DistanceMatrix distanceMatrix = DistanceMatrix.computePairwise(points, distanceMetric);
        int[] medoids = MedoidBuildPhase.selectInitialMedoids(distanceMatrix, targetK, weights);
        NearestMedoidCache medoidCache = new NearestMedoidCache(numPoints);

        double[] slotDeltas = new double[targetK];
        boolean[] isMedoid = new boolean[numPoints];

        int currentIteration = 0;
        boolean hasImproved = true;

        while (hasImproved && currentIteration < maxIterations) {
            medoidCache.recompute(distanceMatrix, medoids);
            SwapMove bestMove = findBestSwap(distanceMatrix, medoids, weights, medoidCache, slotDeltas, isMedoid);
            hasImproved = SwapMove.isImprovement(bestMove);

            if (hasImproved) {
                medoids[bestMove.targetSlot] = bestMove.candidateIdx;
            }
            currentIteration++;
        }

        return DriverLocalKMedoids.modelOf(points, medoids, distanceMetric);
    }

    /** Finds the best (slot, candidate) swap over all non-medoid candidates. */
    private SwapMove findBestSwap(
            DistanceMatrix distanceMatrix,
            int[] medoids,
            double[] pointWeights,
            NearestMedoidCache medoidCache,
            double[] slotDeltas,
            boolean[] isMedoid) {

        Arrays.fill(isMedoid, false);
        for (int medoid : medoids) {
            isMedoid[medoid] = true;
        }

        SwapMove bestMove = SwapMove.NONE;
        for (int candidateIdx = 0; candidateIdx < distanceMatrix.numPoints; candidateIdx++) {
            if (!isMedoid[candidateIdx]) {
                bestMove = SwapMove.getPreferredMove(bestMove,
                    SwapDeltas.findBestMove(distanceMatrix, candidateIdx, pointWeights, medoidCache, slotDeltas));
            }
        }
        return bestMove;
    }
}
