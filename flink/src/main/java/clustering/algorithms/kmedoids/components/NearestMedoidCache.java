package clustering.algorithms.kmedoids.components;

/**
 * Caches per-point distances and slots for the closest and second-closest medoids.
 * Allocated once and updated in place to avoid memory allocations during swap loops.
 */
public final class NearestMedoidCache {

    public final double[] closestDistances;
    public final double[] secondClosestDistances;
    public final int[] closestMedoidSlots;
    public final int[] secondMedoidSlots;

    private final int numPoints;

    public NearestMedoidCache(int numPoints) {
        this.numPoints = numPoints;
        this.closestDistances = new double[numPoints];
        this.secondClosestDistances = new double[numPoints];
        this.closestMedoidSlots = new int[numPoints];
        this.secondMedoidSlots = new int[numPoints];
    }

    /**
     * Recomputes the cached distances and assignments for all points from scratch.
     */
    public void recompute(DistanceMatrix distanceMatrix, int[] medoids) {
        for (int pointIdx = 0; pointIdx < numPoints; pointIdx++) {
            recomputePoint(pointIdx, distanceMatrix, medoids);
        }
    }

    private void recomputePoint(int pointIdx, DistanceMatrix distanceMatrix, int[] medoids) {
        int numMedoids = medoids.length;
        double closestDist = Double.MAX_VALUE;
        double secondClosestDist = Double.MAX_VALUE;
        int closestSlot = -1;
        int secondSlot = -1;

        for (int slot = 0; slot < numMedoids; slot++) {
            double dist = distanceMatrix.getDistance(pointIdx, medoids[slot]);

            if (dist < closestDist) {
                secondClosestDist = closestDist;
                secondSlot = closestSlot;
                closestDist = dist;
                closestSlot = slot;
            } else if (dist < secondClosestDist) {
                secondClosestDist = dist;
                secondSlot = slot;
            }
        }

        closestDistances[pointIdx] = closestDist;
        secondClosestDistances[pointIdx] = secondClosestDist;
        closestMedoidSlots[pointIdx] = closestSlot;
        secondMedoidSlots[pointIdx] = secondSlot;
    }

    /**
     * Updates the cache incrementally after a single medoid swap at {@code swappedSlot}.
     * Performs a constant-time update where possible, falling back to a full point rescan
     * only when the identity of the new second-closest medoid is unknown.
     */
    public void updateAfterSwap(DistanceMatrix distanceMatrix, int[] medoids, int swappedSlot) {
        int newMedoidIdx = medoids[swappedSlot];

        for (int pointIdx = 0; pointIdx < numPoints; pointIdx++) {
            double candidateDist = distanceMatrix.getDistance(pointIdx, newMedoidIdx);
            int nearSlot = closestMedoidSlots[pointIdx];
            int secSlot = secondMedoidSlots[pointIdx];

            if (nearSlot == swappedSlot) {
                if (candidateDist <= secondClosestDistances[pointIdx]) {
                    closestDistances[pointIdx] = candidateDist;
                } else {
                    recomputePoint(pointIdx, distanceMatrix, medoids);
                }
            } else if (secSlot == swappedSlot && candidateDist >= closestDistances[pointIdx]) {
                recomputePoint(pointIdx, distanceMatrix, medoids);
            } else if (candidateDist < closestDistances[pointIdx]) {
                secondClosestDistances[pointIdx] = closestDistances[pointIdx];
                secondMedoidSlots[pointIdx] = nearSlot;
                closestDistances[pointIdx] = candidateDist;
                closestMedoidSlots[pointIdx] = swappedSlot;
            } else if (candidateDist < secondClosestDistances[pointIdx]) {
                secondClosestDistances[pointIdx] = candidateDist;
                secondMedoidSlots[pointIdx] = swappedSlot;
            }
        }
    }
}
