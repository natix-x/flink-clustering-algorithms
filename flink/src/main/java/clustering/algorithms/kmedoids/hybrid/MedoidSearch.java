package clustering.algorithms.kmedoids.hybrid;

import clustering.algorithms.kmedoids.KMedoidsModel;
import clustering.algorithms.kmedoids.components.CandidateSets;
import clustering.algorithms.kmedoids.components.DriverSample;
import clustering.algorithms.kmedoids.components.MedoidIteration;
import clustering.algorithms.kmedoids.components.MedoidRefinement;
import clustering.algorithms.kmedoids.local.DriverLocalKMedoids;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.Points;
import clustering.core.PointSource;
import clustering.core.WeightedPoint;
import clustering.distance.DistanceMetric;
import org.apache.flink.ml.linalg.DenseVector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * The sampling-based medoid search both {@code clara} and {@code pamae} are, as ONE Flink job.
 *
 * <p>CLARA (Kaufman &amp; Rousseeuw 1990) solves an exact k-medoids problem on several small random
 * samples and keeps the medoid set whose cost on the FULL data is lowest. PAMAE (Song et al.,
 * KDD 2017) is that, plus {@code refineIters} Voronoi updates of the winner against the entire data.
 * They differ by exactly one phase — which is what makes the pair a controlled experiment — so here
 * they are one phase machine and {@code refineIters = 0} is CLARA:
 *
 * <pre>
 *   round 0            SAMPLE    draw the numSamples CLARA samples AND (pamae) the candidate pool
 *   round 1            EVALUATE  cost every candidate medoid set against the full data, keep the best
 *   round 2 .. 1+R     REFINE    one Voronoi update of the winner per round, over the full data
 * </pre>
 *
 * <p><b>Why this is one job and used to be three.</b> Every round runs off the per-subtask point
 * cache the iteration fills on its single read of the source, so the passes are free of storage.
 * {@code pamae} previously submitted three separate jobs — CLARA, then a Bernoulli draw for the
 * candidate pool, then the refinement iteration — which on Flink means three reads of the source
 * (there is no cross-job cache) and three job deployments, measured at 4 934 ms of fixed cost each
 * on Ares. On the 196 GB Cohere set the two redundant reads dominate everything the algorithm does.
 * Spark's PAMAE keeps the three-job shape because there it is the cheaper one: its jobs cost a
 * quarter as much to launch and its {@code persist} survives between them. That divergence is the
 * result hard rule 2 asks for, not a defect.
 *
 * <p><b>The pool rides along with the samples.</b> Drawing it in the sampling round is not just one
 * job cheaper, it also fixes a correlation: the pool used to be drawn from
 * {@code new Random(seed + 31·subtask)} walked over the same rows in the same order as CLARA's
 * samples, so at {@code numSamples: 1} the pool was a nested SUPERSET of the seeding sample rather
 * than an independent draw. It is now its own {@link DriverSample.BernoulliStreams} stream.
 *
 * <p><b>Refinement starts from a cost it knows.</b> The standalone {@link MedoidRefinement#refine}
 * has to begin at {@code Double.MAX_VALUE}; here round 1 has just measured the winner's cost on the
 * entire dataset, so the first update has to beat a real number to be accepted.
 */
final class MedoidSearch implements Serializable {

    static final int PHASE_SAMPLE = 0;
    static final int PHASE_EVALUATE = 1;
    static final int PHASE_REFINE = 2;

    /** Rounds fold {@code numSamples} or {@code |pool| + k} doubles per subtask — small enough that
     *  an intermediate merge stage would only add a shuffle. */
    private static final int NO_MERGE_STAGE = 0;

    private final int targetK;
    private final int numSamples;
    private final int sampleSize;
    private final int poolSize;
    private final int refineIters;
    private final DistanceMetric distanceMetric;
    private final long seed;
    private final DriverLocalKMedoids localSolver;

    MedoidSearch(int targetK, int numSamples, int sampleSize, int poolSize, int refineIters,
                 DistanceMetric distanceMetric, DriverLocalKMedoids localSolver, long seed) {
        this.targetK = targetK;
        this.numSamples = numSamples;
        this.sampleSize = sampleSize;
        this.poolSize = poolSize;
        this.refineIters = refineIters;
        this.distanceMetric = distanceMetric;
        this.localSolver = localSolver;
        this.seed = seed;
    }

    /** What the search found: the winner of the sampling phases, and what refinement made of it. */
    static final class Result {
        final KMedoidsModel model;
        /** Cost of the CLARA winner on the entire dataset. */
        final double seedingCost;
        /** Cost after refinement; equal to {@link #seedingCost} when {@code refineIters == 0}. */
        final double finalCost;
        final int refinementRounds;
        final int poolSize;
        final long totalPoints;

        Result(KMedoidsModel model, double seedingCost, double finalCost,
               int refinementRounds, int poolSize, long totalPoints) {
            this.model = model;
            this.seedingCost = seedingCost;
            this.finalCost = finalCost;
            this.refinementRounds = refinementRounds;
            this.poolSize = poolSize;
            this.totalPoints = totalPoints;
        }
    }

    Result run(PointSource source, EnvFactory envFactory) {
        long totalPoints = Datasets.count(source, envFactory);
        if (totalPoints < targetK) {
            throw new IllegalArgumentException(
                "Dataset too small: n=" + totalPoints + " points but k=" + targetK + " requested.");
        }

        // Data that fits in one sample is used whole: one sample drawn with probability 1 IS the
        // dataset, so the small-data case needs no branch of its own — and no extra collect job.
        boolean fitsInOneSample = totalPoints <= sampleSize;
        int effectiveSamples = fitsInOneSample ? 1 : numSamples;
        double sampleFraction = fitsInOneSample
            ? 1.0
            : DriverSample.calculateInclusionProbability(sampleSize, totalPoints);
        double poolFraction = poolSize <= 0
            ? 0.0
            : DriverSample.calculateInclusionProbability(poolSize, totalPoints);

        SearchState initial = new SearchState();
        initial.phase = PHASE_SAMPLE;

        SearchState finalState = MedoidIteration.execute(
            source, envFactory, refineIters > 0 ? "pamae-fit" : "clara-fit",
            initial, SearchState.class, SearchPartial.class,
            new SearchLogic(targetK, effectiveSamples, sampleSize, sampleFraction,
                            poolSize, poolFraction, refineIters, distanceMetric, localSolver, seed),
            NO_MERGE_STAGE);

        if (finalState == null || finalState.bestMedoids == null) {
            throw new IllegalStateException("medoid search produced no medoids (n=" + totalPoints + ")");
        }

        DenseVector[] medoids = new DenseVector[finalState.bestMedoids.length];
        for (int i = 0; i < medoids.length; i++) {
            medoids[i] = Points.wrap(finalState.bestMedoids[i]);
        }
        return new Result(
            new KMedoidsModel(medoids, distanceMetric),
            finalState.seedingCost,
            finalState.bestCost,
            finalState.refine == null ? 0 : finalState.refine.iterationCount,
            finalState.drawnPoolSize,
            totalPoints);
    }

    // ---------------------------------------------------------------------------------------

    /** Broadcast once per round; carries a payload only on the round that has to publish one. */
    public static final class SearchState extends MedoidIteration.State {
        public int phase;
        /** {@code numSamples · k} candidate coordinates, set for the EVALUATE round. */
        public double[][] candidateSets;
        /** Published on the ONE round that enters REFINE; workers memoise it and it is null after. */
        public double[][] pool;
        public int drawnPoolSize;
        public MedoidRefinement.RefineState refine;
        public double[][] bestMedoids;
        public double seedingCost;
        public double bestCost;

        public SearchState() {}

        SearchState copy() {
            SearchState copy = new SearchState();
            copy.phase = phase;
            copy.candidateSets = candidateSets;
            copy.drawnPoolSize = drawnPoolSize;
            copy.refine = refine;
            copy.bestMedoids = bestMedoids;
            copy.seedingCost = seedingCost;
            copy.bestCost = bestCost;
            return copy;
        }
    }

    public static final class SearchPartial extends MedoidIteration.Partial {
        /** SAMPLE round. */
        public double[][] sampledCoords;
        public double[] sampledWeights;
        public int[] sampleIndices;
        public double[][] poolCoords;
        /** EVALUATE and REFINE rounds. */
        public double[] costs;

        public SearchPartial() {}
    }

    private static final class SearchLogic
            implements MedoidIteration.RoundLogic<SearchState, SearchPartial> {

        private final int targetK;
        private final int numSamples;
        private final int sampleSize;
        private final double sampleFraction;
        private final int poolSize;
        private final double poolFraction;
        private final int refineIters;
        private final DistanceMetric distanceMetric;
        private final DriverLocalKMedoids localSolver;
        private final long seed;

        /** Per-subtask memo of the candidate pool, so it is broadcast on ONE round, not every one. */
        private transient double[][] pool;

        SearchLogic(int targetK, int numSamples, int sampleSize, double sampleFraction,
                    int poolSize, double poolFraction, int refineIters,
                    DistanceMetric distanceMetric, DriverLocalKMedoids localSolver, long seed) {
            this.targetK = targetK;
            this.numSamples = numSamples;
            this.sampleSize = sampleSize;
            this.sampleFraction = sampleFraction;
            this.poolSize = poolSize;
            this.poolFraction = poolFraction;
            this.refineIters = refineIters;
            this.distanceMetric = distanceMetric;
            this.localSolver = localSolver;
            this.seed = seed;
        }

        @Override
        public SearchPartial computePartial(
                int round, SearchState state, Iterable<WeightedPoint> points, int subtaskId) {

            SearchPartial partial = new SearchPartial();
            switch (state.phase) {
                case PHASE_SAMPLE:
                    drawSamples(partial, points, subtaskId);
                    return partial;
                case PHASE_EVALUATE:
                    partial.costs = evaluateCandidateSets(state.candidateSets, points);
                    return partial;
                default:
                    if (state.pool != null) {
                        pool = state.pool;
                    }
                    double[][] active = MedoidRefinement.materialize(
                        state.refine.plan, pool, state.refine.medoids);
                    partial.costs = MedoidRefinement.accumulate(
                        points, state.refine.medoids, active, state.refine.plan.clusterBounds, distanceMetric);
                    return partial;
            }
        }

        /** One pass, {@code numSamples} independent CLARA streams plus the candidate-pool stream. */
        private void drawSamples(SearchPartial partial, Iterable<WeightedPoint> points, int subtaskId) {
            boolean drawPool = poolFraction > 0.0;
            int poolStream = numSamples;
            double[] probabilities = new double[drawPool ? numSamples + 1 : numSamples];
            for (int s = 0; s < numSamples; s++) {
                probabilities[s] = sampleFraction;
            }
            if (drawPool) {
                probabilities[poolStream] = poolFraction;
            }
            DriverSample.BernoulliStreams streams =
                new DriverSample.BernoulliStreams(seed, subtaskId, probabilities);

            List<double[]> sampledCoords = new ArrayList<>();
            List<Double> sampledWeights = new ArrayList<>();
            List<Integer> sampleIndices = new ArrayList<>();
            List<double[]> poolCoords = new ArrayList<>();

            for (WeightedPoint point : points) {
                double[] coordinates = point.features.values;
                for (int s = 0; s < numSamples; s++) {
                    if (streams.take(s)) {
                        sampledCoords.add(coordinates.clone());
                        sampledWeights.add(point.weight);
                        sampleIndices.add(s);
                    }
                }
                if (drawPool && streams.take(poolStream)) {
                    poolCoords.add(coordinates.clone());
                }
            }

            partial.sampledCoords = sampledCoords.toArray(new double[0][]);
            partial.sampledWeights = new double[sampledWeights.size()];
            partial.sampleIndices = new int[sampleIndices.size()];
            for (int i = 0; i < partial.sampledWeights.length; i++) {
                partial.sampledWeights[i] = sampledWeights.get(i);
                partial.sampleIndices[i] = sampleIndices.get(i);
            }
            partial.poolCoords = poolCoords.toArray(new double[0][]);
        }

        private double[] evaluateCandidateSets(double[][] candidateSets, Iterable<WeightedPoint> points) {
            CandidateSets.Deduplicated deduplicated = CandidateSets.deduplicate(candidateSets, targetK);
            double[] costs = new double[deduplicated.numSets];
            double[] buffer = deduplicated.newDistanceBuffer();

            for (WeightedPoint point : points) {
                deduplicated.addPointCost(
                    point.features.values, point.weight, distanceMetric, buffer, costs);
            }
            return costs;
        }

        @Override
        public MedoidIteration.Decision<SearchState> combine(
                int round, SearchState state, List<SearchPartial> partials) {

            switch (state.phase) {
                case PHASE_SAMPLE:
                    return afterSampling(state, partials);
                case PHASE_EVALUATE:
                    return afterEvaluation(state, partials);
                default:
                    return afterRefinement(state, partials);
            }
        }

        private MedoidIteration.Decision<SearchState> afterSampling(
                SearchState state, List<SearchPartial> partials) {

            SearchState next = state.copy();
            next.candidateSets = solveLocalSamples(partials);
            next.phase = PHASE_EVALUATE;

            if (poolSize > 0) {
                List<double[]> drawn = new ArrayList<>();
                for (SearchPartial partial : partials) {
                    java.util.Collections.addAll(drawn, partial.poolCoords);
                }
                // The pool truncation gets its own permutation seed, past every sample's.
                pool = DriverSample.takeRandom(drawn, poolSize, seed + numSamples);
                next.drawnPoolSize = pool.length;
            }
            return MedoidIteration.Decision.next(next);
        }

        /** Regroups the subtasks' draws per sample, trims each to {@code sampleSize} and solves it. */
        private double[][] solveLocalSamples(List<SearchPartial> partials) {
            List<List<double[]>> groupedCoords = new ArrayList<>(numSamples);
            List<List<Double>> groupedWeights = new ArrayList<>(numSamples);
            for (int s = 0; s < numSamples; s++) {
                groupedCoords.add(new ArrayList<>());
                groupedWeights.add(new ArrayList<>());
            }

            for (SearchPartial partial : partials) {
                for (int i = 0; i < partial.sampleIndices.length; i++) {
                    groupedCoords.get(partial.sampleIndices[i]).add(partial.sampledCoords[i]);
                    groupedWeights.get(partial.sampleIndices[i]).add(partial.sampledWeights[i]);
                }
            }

            double[][] allCandidates = new double[numSamples * targetK][];
            for (int s = 0; s < numSamples; s++) {
                List<double[]> currentCoords = groupedCoords.get(s);
                if (currentCoords.size() < targetK) {
                    throw new IllegalStateException("Sample " + s + " too small: got " + currentCoords.size()
                        + " points, need at least k=" + targetK + ". Increase sampleSize.");
                }

                int[] shuffleOrder = DriverSample.generateRandomPermutation(currentCoords.size(), seed + s);
                int keepCount = Math.min(sampleSize, currentCoords.size());

                double[][] selectedFeatures = new double[keepCount][];
                double[] selectedWeights = new double[keepCount];
                for (int i = 0; i < keepCount; i++) {
                    int targetIdx = currentCoords.size() <= sampleSize ? i : shuffleOrder[i];
                    selectedFeatures[i] = currentCoords.get(targetIdx);
                    selectedWeights[i] = groupedWeights.get(s).get(targetIdx);
                }

                DenseVector[] localMedoids = localSolver.fitLocal(selectedFeatures, selectedWeights).medoids();
                for (int i = 0; i < targetK; i++) {
                    allCandidates[s * targetK + i] = localMedoids[i].values;
                }
            }
            return allCandidates;
        }

        private MedoidIteration.Decision<SearchState> afterEvaluation(
                SearchState state, List<SearchPartial> partials) {

            double[] totals = sumInOrder(partials);
            int winner = 0;
            for (int set = 1; set < totals.length; set++) {
                if (totals[set] < totals[winner]) {   // ties go to the lowest sample index
                    winner = set;
                }
            }

            SearchState next = state.copy();
            next.bestMedoids = new double[targetK][];
            System.arraycopy(state.candidateSets, winner * targetK, next.bestMedoids, 0, targetK);
            next.seedingCost = totals[winner];
            next.bestCost = totals[winner];
            next.candidateSets = null;

            if (refineIters < 1 || pool == null || pool.length == 0) {
                return MedoidIteration.Decision.stop(next);
            }

            MedoidRefinement.RefineState refine = new MedoidRefinement.RefineState();
            refine.medoids = next.bestMedoids;
            refine.plan = MedoidRefinement.planFor(next.bestMedoids, pool, distanceMetric);
            refine.cost = next.seedingCost;   // a real bar to beat, not Double.MAX_VALUE
            refine.iterationCount = 0;

            next.refine = refine;
            next.phase = PHASE_REFINE;
            next.pool = pool;                 // the one round that publishes it
            return MedoidIteration.Decision.next(next);
        }

        private MedoidIteration.Decision<SearchState> afterRefinement(
                SearchState state, List<SearchPartial> partials) {

            double[] totals = sumInOrder(partials);
            double[][] active = MedoidRefinement.materialize(
                state.refine.plan, pool, state.refine.medoids);
            MedoidRefinement.Outcome outcome = MedoidRefinement.chooseMedoids(
                active, state.refine.plan.clusterBounds, totals, state.refine.medoids);

            MedoidIteration.Decision<MedoidRefinement.RefineState> decision =
                MedoidRefinement.decide(outcome, state.refine, pool, distanceMetric, refineIters);

            SearchState next = state.copy();
            next.refine = decision.nextState;
            next.bestMedoids = decision.nextState.medoids;
            next.bestCost = decision.nextState.cost;
            next.pool = null;                 // every worker memoised it on the entering round

            return decision.stop
                ? MedoidIteration.Decision.stop(next)
                : MedoidIteration.Decision.next(next);
        }

        private static double[] sumInOrder(List<SearchPartial> partials) {
            double[] total = new double[partials.get(0).costs.length];
            for (SearchPartial partial : partials) {
                for (int i = 0; i < total.length; i++) {
                    total[i] += partial.costs[i];
                }
            }
            return total;
        }
    }
}
