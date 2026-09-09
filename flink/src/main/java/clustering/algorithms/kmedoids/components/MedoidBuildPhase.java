package clustering.algorithms.kmedoids.components;

/**
 * BUILD phase (Kaufman & Rousseeuw 1990): Greedy selection of initial k medoids.
 * Shared across local k-medoids implementations to ensure performance differences
 * stem only from the SWAP strategy. Ties resolve to the lowest index.
 */
public final class MedoidBuildPhase {

    private MedoidBuildPhase() {}

    /** Selects indices of the initial k medoids based on point weights and distances. */
    public static int[] selectInitialMedoids(DistanceMatrix distanceMatrix, int targetK, double[] pointWeights) {
        int numPoints = distanceMatrix.numPoints;
        int[] selectedMedoids = new int[targetK];
        boolean[] isMedoidSelected = new boolean[numPoints];

        selectedMedoids[0] = findFirstMedoid(distanceMatrix, pointWeights);
        isMedoidSelected[selectedMedoids[0]] = true;

        double[] nearestMedoidDistances = new double[numPoints];
        for (int i = 0; i < numPoints; i++) {
            nearestMedoidDistances[i] = distanceMatrix.getDistance(selectedMedoids[0], i);
        }

        int medoidsFound = 1;
        while (medoidsFound < targetK) {
            int nextMedoid = findHighestGainMedoid(distanceMatrix, pointWeights, nearestMedoidDistances, isMedoidSelected);
            selectedMedoids[medoidsFound] = nextMedoid;
            isMedoidSelected[nextMedoid] = true;
            medoidsFound++;

            for (int i = 0; i < numPoints; i++) {
                double dist = distanceMatrix.getDistance(nextMedoid, i);
                if (dist < nearestMedoidDistances[i]) {
                    nearestMedoidDistances[i] = dist;
                }
            }
        }
        return selectedMedoids;
    }

    /** Finds the point minimizing the total weighted distance to all other points. */
    private static int findFirstMedoid(DistanceMatrix distanceMatrix, double[] pointWeights) {
        int numPoints = distanceMatrix.numPoints;
        int bestIndex = -1;
        double minTotalDistance = Double.MAX_VALUE;

        for (int candidateIdx = 0; candidateIdx < numPoints; candidateIdx++) {
            double currentTotal = 0.0;
            for (int targetIdx = 0; targetIdx < numPoints; targetIdx++) {
                currentTotal += pointWeights[targetIdx] * distanceMatrix.getDistance(candidateIdx, targetIdx);
            }
            if (currentTotal < minTotalDistance) {
                minTotalDistance = currentTotal;
                bestIndex = candidateIdx;
            }
        }
        return bestIndex;
    }

    /** Finds the unselected point that maximizes the weighted cost reduction. */
    private static int findHighestGainMedoid(
            DistanceMatrix distanceMatrix,
            double[] pointWeights,
            double[] nearestMedoidDistances,
            boolean[] isMedoidSelected) {

        int numPoints = distanceMatrix.numPoints;
        int bestIndex = -1;
        double maxGain = Double.NEGATIVE_INFINITY;

        for (int candidateIdx = 0; candidateIdx < numPoints; candidateIdx++) {
            if (isMedoidSelected[candidateIdx]) {
                continue;
            }
            double currentGain = 0.0;
            for (int targetIdx = 0; targetIdx < numPoints; targetIdx++) {
                double distanceReduction = nearestMedoidDistances[targetIdx] - distanceMatrix.getDistance(candidateIdx, targetIdx);
                if (distanceReduction > 0.0) {
                    currentGain += pointWeights[targetIdx] * distanceReduction;
                }
            }
            if (currentGain > maxGain) {
                maxGain = currentGain;
                bestIndex = candidateIdx;
            }
        }
        return bestIndex;
    }
}
