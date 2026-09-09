package clustering.algorithms.kmedoids.components;

/**
 * Represents replacing a medoid in a specific slot with a candidate point.
 * Tracks the resulting change in the objective function (cost delta).
 */
public final class SwapMove {

    /** Neutral move indicating no improvement. */
    public static final SwapMove NONE = new SwapMove(0.0, -1, -1);

    public final double costDelta;
    public final int targetSlot;
    public final int candidateIdx;

    public SwapMove(double costDelta, int targetSlot, int candidateIdx) {
        this.costDelta = costDelta;
        this.targetSlot = targetSlot;
        this.candidateIdx = candidateIdx;
    }

    /**
     * Determines the better swap move.
     * Lower cost delta wins. Ties resolve to the lower slot, then lower candidate index
     * to guarantee deterministic results across all algorithm implementations.
     */
    public static SwapMove getPreferredMove(SwapMove moveA, SwapMove moveB) {
        if (moveB.targetSlot < 0) {
            return moveA;
        }
        if (moveA.targetSlot < 0) {
            return moveB;
        }
        if (moveA.costDelta != moveB.costDelta) {
            return moveA.costDelta < moveB.costDelta ? moveA : moveB;
        }
        if (moveA.targetSlot != moveB.targetSlot) {
            return moveA.targetSlot < moveB.targetSlot ? moveA : moveB;
        }
        return moveA.candidateIdx <= moveB.candidateIdx ? moveA : moveB;
    }

    public static boolean isImprovement(SwapMove move) {
        return move.targetSlot >= 0 && move.costDelta < 0.0;
    }
}
