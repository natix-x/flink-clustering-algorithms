package clustering.algorithms.kmedoids.distributed;

import clustering.algorithms.kmedoids.KMedoidsModel;
import clustering.algorithms.kmedoids.components.MedoidIteration;
import clustering.algorithms.kmedoids.components.SwapMove;
import clustering.core.Clusterer;
import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.core.Points;
import clustering.core.WeightedPoint;
import clustering.distance.DistanceMetric;
import org.apache.flink.ml.linalg.DenseVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Distributed FastPAM1 (BUILD + SWAP) as a single Flink ML bounded-iteration job.
 *
 * <p>Every point is a candidate medoid, so a round's accumulator is the FastPAM1 decomposition
 * {@code shared[n] + removeLoss[n·k]} and one round applies the single best improving swap.
 *
 * <h3>The candidates are gathered by the job, not handed to it</h3>
 * The candidate array used to be built by a {@code Datasets.collectAll} before the job: a SECOND
 * full read of the source, taken at parallelism 1, whose only purpose was to put {@code n·d} doubles
 * in the operator's closure — from where Flink shipped them to every subtask anyway. Round 0 now
 * gathers them from the point cache instead and publishes them on the feedback edge, so the
 * broadcast volume is unchanged and the extra read and the single-threaded collect are gone.
 *
 * <p>The price is that candidate ORDER becomes subtask-major rather than source order, which moves
 * tie-breaking. Run-to-run reproducibility is untouched (partials are folded in ascending subtask
 * order), and cross-engine identity was never available here anyway — Spark's candidate array is in
 * partition order, which is not the Flink source order either.
 *
 * <h3>Why this one folds through a tree</h3>
 * Its partial is {@code n·(k+1)} doubles — 880 KB per subtask at n = 10 000, k = 10, i.e. 56 MB per
 * round arriving at ONE task at parallelism 64, for every BUILD and SWAP round of the run. The
 * iteration therefore runs with a merge stage: contiguous groups of {@link #MERGE_FAN_IN} subtasks
 * are folded first, cutting what reaches the driver by that factor. Groups are subtask RANGES and a
 * group is never split across merge tasks, so the fold order is still fixed by slot index and the
 * result is bit-identical to the flat fold.
 */
public class DistributedFastPAM implements Clusterer {

    private static final int PHASE_GATHER = 0;
    private static final int PHASE_SEARCH = 1;

    /** Subtasks per pre-fold group. See the class doc. */
    private static final int MERGE_FAN_IN = 8;

    private final int targetK;
    private final int maxIterations;
    private final DistanceMetric distanceMetric;

    public DistributedFastPAM(int targetK, int maxIterations, DistanceMetric distanceMetric) {
        this.targetK = targetK;
        this.maxIterations = maxIterations;
        this.distanceMetric = distanceMetric;
    }

    @Override
    public Model fit(PointSource source, EnvFactory envFactory, int parallelism) {
        FastPamState initial = new FastPamState();
        initial.phase = PHASE_GATHER;

        FastPamState finalState = MedoidIteration.execute(
            source, envFactory, "distfastpam-fit",
            initial, FastPamState.class, FastPamPartial.class,
            new FastPamLogic(targetK, maxIterations, distanceMetric),
            MERGE_FAN_IN);

        if (finalState == null || finalState.bestMedoids == null) {
            throw new IllegalStateException("distfastpam produced no medoids");
        }

        DenseVector[] medoidVectors = new DenseVector[targetK];
        for (int i = 0; i < targetK; i++) {
            medoidVectors[i] = Points.wrap(finalState.bestMedoids[i]);
        }
        return new KMedoidsModel(medoidVectors, distanceMetric);
    }

    public static final class FastPamState extends MedoidIteration.State {
        public int phase;
        /** All candidates; published on the ONE round that enters the search, null afterwards. */
        public double[][] candidates;
        public int[] medoids;
        public int numSelected;
        public int completedSwapRounds;
        public double[][] bestMedoids;

        public FastPamState() {}

        FastPamState copy() {
            FastPamState copy = new FastPamState();
            copy.phase = phase;
            copy.medoids = medoids == null ? null : medoids.clone();
            copy.numSelected = numSelected;
            copy.completedSwapRounds = completedSwapRounds;
            copy.bestMedoids = bestMedoids;
            return copy;
        }
    }

    public static final class FastPamPartial extends MedoidIteration.Partial {
        /** GATHER round: this subtask's points, in cache order. */
        public double[][] gathered;
        /** BUILD / SWAP round: the FastPAM1 accumulator. */
        public double[] sums;

        public FastPamPartial() {}
    }

    private static final class FastPamLogic
            implements MedoidIteration.RoundLogic<FastPamState, FastPamPartial> {

        private final int targetK;
        private final int maxIterations;
        private final DistanceMetric distanceMetric;

        /** Per-subtask memo, so the candidates are broadcast once and not once per round. */
        private transient double[][] candidates;

        FastPamLogic(int targetK, int maxIterations, DistanceMetric distanceMetric) {
            this.targetK = targetK;
            this.maxIterations = maxIterations;
            this.distanceMetric = distanceMetric;
        }

        @Override
        public FastPamPartial computePartial(
                int round, FastPamState state, Iterable<WeightedPoint> points, int subtaskId) {

            FastPamPartial partial = new FastPamPartial();
            if (state.phase == PHASE_GATHER) {
                List<double[]> gathered = new ArrayList<>();
                for (WeightedPoint point : points) {
                    // Cached records are reused between rounds, so the coordinates are copied out.
                    gathered.add(point.features.values.clone());
                }
                partial.gathered = gathered.toArray(new double[0][]);
                return partial;
            }

            if (state.candidates != null) {
                candidates = state.candidates;
            }
            partial.sums = state.numSelected < targetK
                ? aggregateBuildPhase(state, points)
                : aggregateSwapPhase(state, points);
            return partial;
        }

        private double[] aggregateBuildPhase(FastPamState state, Iterable<WeightedPoint> points) {
            int numCandidates = candidates.length;
            double[] accumulator = new double[numCandidates];

            if (state.numSelected == 0) {
                for (WeightedPoint point : points) {
                    double[] coords = point.features.values;
                    double weight = point.weight;
                    for (int i = 0; i < numCandidates; i++) {
                        accumulator[i] += weight * distanceMetric.compute(candidates[i], coords);
                    }
                }
                return accumulator;
            }

            double[][] selectedMedoids = extractSelected(candidates, state.medoids, state.numSelected);
            for (WeightedPoint point : points) {
                double[] coords = point.features.values;
                double weight = point.weight;
                double minDistance = Double.MAX_VALUE;

                for (double[] medoid : selectedMedoids) {
                    double dist = distanceMetric.compute(medoid, coords);
                    if (dist < minDistance) {
                        minDistance = dist;
                    }
                }
                for (int i = 0; i < numCandidates; i++) {
                    double gain = minDistance - distanceMetric.compute(candidates[i], coords);
                    if (gain > 0.0) {
                        accumulator[i] += weight * gain;
                    }
                }
            }
            return accumulator;
        }

        private double[] aggregateSwapPhase(FastPamState state, Iterable<WeightedPoint> points) {
            int numCandidates = candidates.length;
            double[][] activeMedoids = extractSelected(candidates, state.medoids, targetK);
            double[] accumulator = new double[numCandidates + numCandidates * targetK];

            for (WeightedPoint point : points) {
                double[] coords = point.features.values;
                double weight = point.weight;
                double closestDist = Double.MAX_VALUE;
                double secondClosestDist = Double.MAX_VALUE;
                int closestMedoidIdx = -1;

                for (int m = 0; m < targetK; m++) {
                    double dist = distanceMetric.compute(activeMedoids[m], coords);
                    if (dist < closestDist) {
                        secondClosestDist = closestDist;
                        closestDist = dist;
                        closestMedoidIdx = m;
                    } else if (dist < secondClosestDist) {
                        secondClosestDist = dist;
                    }
                }

                for (int i = 0; i < numCandidates; i++) {
                    double candidateDist = distanceMetric.compute(candidates[i], coords);
                    double sharedContribution = candidateDist < closestDist ? weight * (candidateDist - closestDist) : 0.0;
                    accumulator[i] += sharedContribution;

                    double removeLoss = weight * (Math.min(secondClosestDist, candidateDist) - closestDist);
                    accumulator[numCandidates + i * targetK + closestMedoidIdx] += (removeLoss - sharedContribution);
                }
            }
            return accumulator;
        }

        @Override
        public FastPamPartial merge(FastPamPartial left, FastPamPartial right) {
            if (left.gathered != null) {
                double[][] joined = new double[left.gathered.length + right.gathered.length][];
                System.arraycopy(left.gathered, 0, joined, 0, left.gathered.length);
                System.arraycopy(right.gathered, 0, joined, left.gathered.length, right.gathered.length);
                left.gathered = joined;
                return left;
            }
            for (int i = 0; i < left.sums.length; i++) {
                left.sums[i] += right.sums[i];
            }
            return left;
        }

        @Override
        public MedoidIteration.Decision<FastPamState> combine(
                int round, FastPamState state, List<FastPamPartial> partials) {

            if (state.phase == PHASE_GATHER) {
                return afterGather(state, partials);
            }
            return afterSearchRound(state, partials);
        }

        private MedoidIteration.Decision<FastPamState> afterGather(
                FastPamState state, List<FastPamPartial> partials) {

            List<double[]> collected = new ArrayList<>();
            for (FastPamPartial partial : partials) {
                java.util.Collections.addAll(collected, partial.gathered);
            }
            if (collected.size() < targetK) {
                throw new IllegalArgumentException("Dataset too small: n=" + collected.size()
                    + " points but k=" + targetK + " medoids requested.");
            }
            candidates = collected.toArray(new double[0][]);

            FastPamState next = state.copy();
            next.phase = PHASE_SEARCH;
            next.candidates = candidates;
            next.medoids = new int[targetK];
            Arrays.fill(next.medoids, -1);
            next.numSelected = 0;
            next.completedSwapRounds = 0;
            return MedoidIteration.Decision.next(next);
        }

        private MedoidIteration.Decision<FastPamState> afterSearchRound(
                FastPamState state, List<FastPamPartial> partials) {

            int numCandidates = candidates.length;
            double[] globalSums = new double[partials.get(0).sums.length];
            for (FastPamPartial partial : partials) {
                for (int i = 0; i < globalSums.length; i++) {
                    globalSums[i] += partial.sums[i];
                }
            }

            FastPamState next = state.copy();
            next.candidates = null;   // every worker memoised it on the entering round
            boolean stop;

            if (state.numSelected < targetK) {
                boolean[] excluded = new boolean[numCandidates];
                for (int i = 0; i < state.numSelected; i++) {
                    excluded[state.medoids[i]] = true;
                }
                int nextIndex = (state.numSelected == 0)
                    ? findMinExcluding(globalSums, excluded)
                    : findMaxExcluding(globalSums, excluded);

                next.medoids[state.numSelected] = nextIndex;
                next.numSelected = state.numSelected + 1;
                stop = false;
            } else {
                boolean[] isMedoid = new boolean[numCandidates];
                for (int i = 0; i < targetK; i++) {
                    isMedoid[state.medoids[i]] = true;
                }

                SwapMove bestMove = SwapMove.NONE;
                for (int candidateIdx = 0; candidateIdx < numCandidates; candidateIdx++) {
                    if (isMedoid[candidateIdx]) {
                        continue;
                    }
                    double sharedGain = globalSums[candidateIdx];
                    int baseIndex = numCandidates + candidateIdx * targetK;

                    for (int slot = 0; slot < targetK; slot++) {
                        double delta = sharedGain + globalSums[baseIndex + slot];
                        if (delta < 0.0) {
                            bestMove = SwapMove.getPreferredMove(bestMove, new SwapMove(delta, slot, candidateIdx));
                        }
                    }
                }
                next.completedSwapRounds = state.completedSwapRounds + 1;
                boolean isImprovement = SwapMove.isImprovement(bestMove);
                if (isImprovement) {
                    next.medoids[bestMove.targetSlot] = bestMove.candidateIdx;
                }
                stop = !isImprovement || next.completedSwapRounds >= maxIterations;
            }

            if (stop) {
                next.bestMedoids = extractSelected(candidates, next.medoids, targetK);
            }
            return stop
                ? MedoidIteration.Decision.stop(next)
                : MedoidIteration.Decision.next(next);
        }
    }

    private static double[][] extractSelected(double[][] candidates, int[] indices, int count) {
        double[][] selected = new double[count][];
        for (int i = 0; i < count; i++) {
            selected[i] = candidates[indices[i]];
        }
        return selected;
    }

    private static int findMinExcluding(double[] values, boolean[] excluded) {
        int bestIdx = -1;
        double minValue = Double.MAX_VALUE;
        for (int i = 0; i < values.length; i++) {
            if (!excluded[i] && values[i] < minValue) {
                minValue = values[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    private static int findMaxExcluding(double[] values, boolean[] excluded) {
        int bestIdx = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            if (!excluded[i] && values[i] > maxValue) {
                maxValue = values[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
}
