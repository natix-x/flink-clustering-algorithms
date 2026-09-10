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
 * The sampling-based medoid search for CLARA and PAMAE, executed as a single Flink job.
 *
 * Executes up to three phases:
 * 1. SAMPLE: Draws random samples (and optionally the candidate pool).
 * 2. EVALUATE: Evaluates every candidate medoid set against the full dataset and selects the best.
 * 3. REFINE: Performs Voronoi updates of the winning set over the full data (if refineIters > 0).
 */
final class MedoidSearch implements Serializable {

    static final int PHASE_SAMPLE = 0;
    static final int PHASE_EVALUATE = 1;
    static final int PHASE_REFINE = 2;

    /** No intermediate merge stage is needed as rounds fold small arrays of doubles. */
    private static final int NO_MERGE_STAGE = 0;

    private final int targetK;
    private final int numSamples;
    private final int sampleSize;
    private final int poolSize;
    private final int refineIters;
    private final DistanceMetric distanceMetric;
    private final long seed;
    private final DriverLocalKMedoids localSolver;

    MedoidSearch(
            int targetK,
            int numSamples,
            int sampleSize,
            int poolSize,
            int refineIters,
            DistanceMetric distanceMetric,
            DriverLocalKMedoids localSolver,
            long seed
    ) {
        this.targetK = targetK;
        this.numSamples = numSamples;
        this.sampleSize = sampleSize;
        this.poolSize = poolSize;
        this.refineIters = refineIters;
        this.distanceMetric = distanceMetric;
        this.localSolver = localSolver;
        this.seed = seed;
    }

    /**
     * Encapsulates the winning model and cost metrics from the sampling and refinement phases.
     */
    static final class Result {
        final KMedoidsModel model;
        final double seedingCost;
        final double finalCost;
        final int refinementRounds;
        final int poolSize;
        final long totalPoints;

        Result(
                KMedoidsModel model,
                double seedingCost,
                double finalCost,
                int refinementRounds,
                int poolSize,
                long totalPoints
        ) {
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
                "Dataset too small: n=" + totalPoints + " points but k=" + targetK + " requested."
            );
        }

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
            new SearchLogic(
                targetK, effectiveSamples, sampleSize, sampleFraction,
                poolSize, poolFraction, refineIters, distanceMetric, localSolver, seed
            ),
            NO_MERGE_STAGE
        );

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
            totalPoints
        );
    }

    // ---------------------------------------------------------------------------------------

    public static final class SearchState extends MedoidIteration.State {
        public int phase;
        public double[][] candidateSets;
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
        public double[][] sampledCoords;
        public double[] sampledWeights;
        public int[] sampleIndices;
        public double[][] poolCoords;
        public double[] costs;

        public SearchPartial() {}
    }

    private static final class SearchLogic implements MedoidIteration.RoundLogic<SearchState, SearchPartial> {

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

        private transient double[][] pool;

        SearchLogic(
                int targetK, int numSamples, int sampleSize, double sampleFraction,
                int poolSize, double poolFraction, int refineIters,
                DistanceMetric distanceMetric, DriverLocalKMedoids localSolver, long seed
        ) {
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
                int round, SearchState state, Iterable<WeightedPoint> points, int subtaskId
        ) {
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
                        state.refine.plan, pool, state.refine.medoids
                    );
                    partial.costs = MedoidRefinement.accumulate(
                        points, state.refine.medoids, active, state.refine.plan.clusterBounds, distanceMetric
                    );
                    return partial;
            }
        }

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

            DriverSample.BernoulliStreams streams = new DriverSample.BernoulliStreams(seed, subtaskId, probabilities);

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
                deduplicated.addPointCost(point.features.values, point.weight, distanceMetric, buffer, costs);
            }
            return costs;
        }

        @Override
        public MedoidIteration.Decision<SearchState> combine(
                int round, SearchState state, List<SearchPartial> partials
        ) {
            switch (state.phase) {
                case PHASE_SAMPLE:
                    return afterSampling(state, partials);
                case PHASE_EVALUATE:
                    return afterEvaluation(state, partials);
                default:
                    return afterRefinement(state, partials);
            }
        }

        private MedoidIteration.Decision<SearchState> afterSampling(SearchState state, List<SearchPartial> partials) {
            SearchState next = state.copy();
            next.candidateSets = solveLocalSamples(partials);
            next.phase = PHASE_EVALUATE;

            if (poolSize > 0) {
                List<double[]> drawn = new ArrayList<>();
                for (SearchPartial partial : partials) {
                    java.util.Collections.addAll(drawn, partial.poolCoords);
                }
                pool = DriverSample.takeRandom(drawn, poolSize, seed + numSamples);
                next.drawnPoolSize = pool.length;
            }
            return MedoidIteration.Decision.next(next);
        }

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
                    throw new IllegalStateException(
                        "Sample " + s + " too small: got " + currentCoords.size() +
                        " points, need at least k=" + targetK + ". Increase sampleSize."
                    );
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

        private MedoidIteration.Decision<SearchState> afterEvaluation(SearchState state, List<SearchPartial> partials) {
            double[] totals = sumInOrder(partials);
            int winner = 0;

            for (int set = 1; set < totals.length; set++) {
                if (totals[set] < totals[winner]) {
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
            refine.cost = next.seedingCost;
            refine.iterationCount = 0;

            next.refine = refine;
            next.phase = PHASE_REFINE;
            next.pool = pool;

            return MedoidIteration.Decision.next(next);
        }

        private MedoidIteration.Decision<SearchState> afterRefinement(SearchState state, List<SearchPartial> partials) {
            double[] totals = sumInOrder(partials);
            double[][] active = MedoidRefinement.materialize(state.refine.plan, pool, state.refine.medoids);

            MedoidRefinement.Outcome outcome = MedoidRefinement.chooseMedoids(
                active, state.refine.plan.clusterBounds, totals, state.refine.medoids
            );

            MedoidIteration.Decision<MedoidRefinement.RefineState> decision =
                MedoidRefinement.decide(outcome, state.refine, pool, distanceMetric, refineIters);

            SearchState next = state.copy();
            next.refine = decision.nextState;
            next.bestMedoids = decision.nextState.medoids;
            next.bestCost = decision.nextState.cost;
            next.pool = null;

            return decision.stop ? MedoidIteration.Decision.stop(next) : MedoidIteration.Decision.next(next);
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
