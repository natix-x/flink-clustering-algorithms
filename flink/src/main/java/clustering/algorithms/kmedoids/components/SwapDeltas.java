package clustering.algorithms.kmedoids.components;

import clustering.algorithms.kmedoids.local.FastPAM;
import clustering.algorithms.kmedoids.local.FasterPAM;
import java.util.Arrays;

/**
 * Computes FastPAM1 swap cost changes in a single O(n) pass.
 * Evaluates the cost change for swapping a candidate into every medoid slot simultaneously.
 */
public final class SwapDeltas {

    private SwapDeltas() {}

    /** Computes swap cost deltas for a candidate across all medoid slots. */
    public static void calculateSlotDeltas(
            DistanceMatrix distanceMatrix,
            int candidateIdx,
            double[] pointWeights,
            NearestMedoidCache medoidCache,
            double[] slotDeltas) {

        Arrays.fill(slotDeltas, 0.0);

        double totalSharedGain = 0.0;
        int numPoints = distanceMatrix.numPoints;

        for (int i = 0; i < numPoints; i++) {
            double candidateDist = distanceMatrix.getDistance(candidateIdx, i);
            double weight = pointWeights[i];
            double closestDist = medoidCache.closestDistances[i];

            double pointSharedGain = candidateDist < closestDist ? weight * (candidateDist - closestDist) : 0.0;
            totalSharedGain += pointSharedGain;

            slotDeltas[medoidCache.closestMedoidSlots[i]] +=
                weight * (Math.min(medoidCache.secondClosestDistances[i], candidateDist) - closestDist)
                    - pointSharedGain;
        }

        for (int slot = 0; slot < slotDeltas.length; slot++) {
            slotDeltas[slot] += totalSharedGain;
        }
    }

    /** Finds the best improving swap move for the candidate, or SwapMove.NONE if none exists. */
    public static SwapMove findBestMove(
            DistanceMatrix distanceMatrix,
            int candidateIdx,
            double[] pointWeights,
            NearestMedoidCache medoidCache,
            double[] slotDeltas) {

        calculateSlotDeltas(distanceMatrix, candidateIdx, pointWeights, medoidCache, slotDeltas);
        SwapMove bestMove = SwapMove.NONE;

        for (int slot = 0; slot < slotDeltas.length; slot++) {
            if (slotDeltas[slot] < 0.0) {
                bestMove = SwapMove.getPreferredMove(bestMove, new SwapMove(slotDeltas[slot], slot, candidateIdx));
            }
        }
        return bestMove;
    }
}
