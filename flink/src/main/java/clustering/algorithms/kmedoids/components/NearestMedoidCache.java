package clustering.algorithms.kmedoids.components;

/**
 * Caches per-point distances to the closest and second-closest medoids, plus the SLOT (0..k-1)
 * of each — the four arrays the FastPAM1 Δ formula ({@link SwapDeltas}) and FasterPAM's
 * incremental update both work over.
 *
 * <p>Allocated once per fit and updated in place, so the swap loops allocate nothing.
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
     * Recomputes the cached distances and assignments for all points, from scratch.
     * Runs in O(n·k) time to guarantee an exact cache state — used for the initial fill and as
     * the rescan fallback inside {@link #updateAfterSwap}.
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
     * Updates the cache for ONE swap at {@code swappedSlot} — {@code medoids[swappedSlot]} is
     * already the NEW medoid — in O(n) plus a full O(k) rescan for whichever points had the old
     * medoid in their own top 2. Every other point is handled in O(1), by the same three-way split
     * the FastPAM1/FasterPAM papers use:
     *
     * <ul>
     *   <li>{@code swappedSlot} was neither this point's nearest nor second-nearest — inserting
     *       the new medoid at its distance {@code doj} is a plain top-2 update, and it is exact:
     *       nothing else in the point's top 2 moved.</li>
     *   <li>{@code swappedSlot} WAS the nearest, and {@code doj} is still within the old second's
     *       distance — the new medoid simply takes over that slot as nearest; the second is
     *       unaffected because the medoid it names did not move. Still O(1) and exact.</li>
     *   <li>Otherwise (the old nearest or second is being displaced by something the cache cannot
     *       rank against a THIRD medoid it never tracked) a full rescan is the only exact answer,
     *       so it is done — for that point only.</li>
     * </ul>
     *
     * <p>This is what {@code FasterPAM}'s docs promise but the previous implementation did not
     * deliver: a real O(n) amortised update rather than an O(n·k) recompute after every accepted
     * swap. Whether a point falls into the rescan case depends only on which medoid it was
     * assigned to, so the total rescan work across a pass is bounded by (roughly) the size of the
     * cluster the swapped-out medoid served — O(n) in total when clusters are balanced, same as
     * the paper's stated amortised cost, and never worse than the O(n·k) this replaces since a
     * rescan only ever touches the fraction of points that needed it.
     */
    public void updateAfterSwap(DistanceMatrix distanceMatrix, int[] medoids, int swappedSlot) {
        int newMedoidIdx = medoids[swappedSlot];

        for (int pointIdx = 0; pointIdx < numPoints; pointIdx++) {
            double candidateDist = distanceMatrix.getDistance(pointIdx, newMedoidIdx);
            int nearSlot = closestMedoidSlots[pointIdx];
            int secSlot = secondMedoidSlots[pointIdx];

            if (nearSlot == swappedSlot) {
                if (candidateDist <= secondClosestDistances[pointIdx]) {
                    // The new medoid still owns this slot as nearest; the second is untouched
                    // because the medoid it names never moved. Exact, O(1).
                    closestDistances[pointIdx] = candidateDist;
                } else {
                    // The old second is now the true nearest, but the new second is a medoid this
                    // cache never tracked (the point's "third nearest") — only a rescan is exact.
                    recomputePoint(pointIdx, distanceMatrix, medoids);
                }
            } else if (secSlot == swappedSlot && candidateDist >= closestDistances[pointIdx]) {
                // The old second is being displaced and the candidate doesn't even beat the
                // nearest, so whether it becomes the new second depends on a third medoid this
                // cache never tracked. Rescan.
                recomputePoint(pointIdx, distanceMatrix, medoids);
            } else if (candidateDist < closestDistances[pointIdx]) {
                // swappedSlot was not in the top 2 (or was the second and just got promoted past
                // the nearest) — inserting it as the new nearest and demoting the old nearest to
                // second is exact either way.
                secondClosestDistances[pointIdx] = closestDistances[pointIdx];
                secondMedoidSlots[pointIdx] = nearSlot;
                closestDistances[pointIdx] = candidateDist;
                closestMedoidSlots[pointIdx] = swappedSlot;
            } else if (candidateDist < secondClosestDistances[pointIdx]) {
                secondClosestDistances[pointIdx] = candidateDist;
                secondMedoidSlots[pointIdx] = swappedSlot;
            }
            // else: swappedSlot is outside this point's top 2 both before and after — unchanged.
        }
    }
}
