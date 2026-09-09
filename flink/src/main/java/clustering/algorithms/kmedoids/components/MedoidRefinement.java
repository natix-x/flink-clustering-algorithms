package clustering.algorithms.kmedoids.components;

import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.Points;
import clustering.core.PointSource;
import clustering.core.WeightedPoint;
import clustering.distance.DistanceMetric;
import org.apache.flink.ml.linalg.DenseVector;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * Distributed medoid update over the entire dataset — phase II of PAMAE (Song et al., KDD 2017).
 *
 * <p>One round is one Voronoi update: assign every point to its nearest current medoid, and for
 * every cluster pick the candidate from the pool that minimises the weighted distance sum WITHIN
 * that cluster. Because a candidate only ever competes inside the cluster it belongs to, one round
 * costs {@code k + |pool| / k} distance computations per point rather than {@code |pool|}.
 *
 * <p>The class is split in two on purpose. The static half ({@link #planFor}, {@link #accumulate},
 * {@link #chooseMedoids}) is the refinement MATHS and has no Flink in it; {@link #refine} is the
 * standalone entry point that runs those rounds as one bounded iteration. {@code pamae} uses the
 * static half directly, as the tail phases of the single job that also does its CLARA seeding — so
 * the two paths cannot drift, and the specs that exercise {@link #refine} are exercising the code
 * {@code pamae} actually runs.
 */
public final class MedoidRefinement {

    /** Rounds of one refinement job are cheap folds of {@code |pool| + k} doubles, so there is
     *  nothing for an intermediate merge stage to save. */
    private static final int NO_MERGE_STAGE = 0;

    private MedoidRefinement() {}

    /** Result of the refinement process containing the final medoids and cost. */
    public static final class Result {
        public final DenseVector[] medoids;
        public final double cost;
        public final int iterations;

        public Result(DenseVector[] medoids, double cost, int iterations) {
            this.medoids = medoids;
            this.cost = cost;
            this.iterations = iterations;
        }
    }

    // ---------------------------------------------------------------------------------------
    // The refinement maths — shared with the merged clara/pamae search, no Flink involved.
    // ---------------------------------------------------------------------------------------

    /** Which candidates compete in which cluster, given the current medoids.
     *
     *  <p>The incumbent medoids are appended to the pool, so a slot can always re-elect what it
     *  already holds — which is what makes a round unable to increase the objective. */
    public static final class Plan implements Serializable {
        /** Candidate indices (pool first, then the k incumbents), grouped by cluster. */
        public int[] candidateOrder;
        /** {@code clusterBounds[c] .. clusterBounds[c+1]} is cluster c's slice of the order. */
        public int[] clusterBounds;

        /** Public and mutable, with a no-arg constructor, because it is a field of a state that
         *  crosses the iteration's feedback edge: anything Flink cannot read as a POJO falls back
         *  to Kryo, which on a JDK 17 build fails to initialise at all. */
        public Plan() {}

        Plan(int[] candidateOrder, int[] clusterBounds) {
            this.candidateOrder = candidateOrder;
            this.clusterBounds = clusterBounds;
        }
    }

    /** Groups pool ∪ medoids by the cluster each candidate falls in. */
    public static Plan planFor(double[][] medoids, double[][] pool, DistanceMetric distanceMetric) {
        int numClusters = medoids.length;
        int totalCandidates = pool.length + numClusters;
        int[] clusterAssignment = new int[totalCandidates];
        int[] bounds = new int[numClusters + 1];

        for (int i = 0; i < totalCandidates; i++) {
            double[] candidate = i < pool.length ? pool[i] : medoids[i - pool.length];
            int cluster = findNearestCluster(candidate, medoids, distanceMetric);
            clusterAssignment[i] = cluster;
            bounds[cluster + 1]++;
        }
        for (int i = 0; i < numClusters; i++) {
            bounds[i + 1] += bounds[i];
        }

        int[] cursor = bounds.clone();
        int[] order = new int[totalCandidates];
        for (int i = 0; i < totalCandidates; i++) {
            order[cursor[clusterAssignment[i]]++] = i;
        }
        return new Plan(order, bounds);
    }

    /** The candidates of a {@link Plan}, in plan order. */
    public static double[][] materialize(Plan plan, double[][] pool, double[][] medoids) {
        double[][] materialized = new double[plan.candidateOrder.length][];
        for (int i = 0; i < plan.candidateOrder.length; i++) {
            int index = plan.candidateOrder[i];
            materialized[i] = index < pool.length ? pool[index] : medoids[index - pool.length];
        }
        return materialized;
    }

    /** One subtask's share of the per-candidate cost sums. */
    public static double[] accumulate(
            Iterable<WeightedPoint> points,
            double[][] medoids,
            double[][] activeCandidates,
            int[] clusterBounds,
            DistanceMetric distanceMetric) {

        double[] costs = new double[activeCandidates.length];
        for (WeightedPoint point : points) {
            double[] coordinates = point.features.values;
            double weight = point.weight;
            int cluster = findNearestCluster(coordinates, medoids, distanceMetric);

            for (int i = clusterBounds[cluster]; i < clusterBounds[cluster + 1]; i++) {
                costs[i] += weight * distanceMetric.compute(coordinates, activeCandidates[i]);
            }
        }
        return costs;
    }

    /** The medoid set a round elects, and whether it actually moved. */
    public static final class Outcome {
        public final double[][] medoids;
        public final double cost;
        public final boolean changed;

        Outcome(double[][] medoids, double cost, boolean changed) {
            this.medoids = medoids;
            this.cost = cost;
            this.changed = changed;
        }
    }

    /** Picks each cluster's cheapest candidate from the globally summed costs. */
    public static Outcome chooseMedoids(
            double[][] activeCandidates,
            int[] clusterBounds,
            double[] candidateCosts,
            double[][] currentMedoids) {

        int numClusters = currentMedoids.length;
        double[][] updated = new double[numClusters][];
        double updatedCost = 0.0;

        for (int cluster = 0; cluster < numClusters; cluster++) {
            int best = findBestCandidateForCluster(
                cluster, activeCandidates, clusterBounds, candidateCosts, currentMedoids);
            if (best < 0) {
                updated[cluster] = currentMedoids[cluster];
            } else {
                updated[cluster] = activeCandidates[best];
                updatedCost += candidateCosts[best];
            }
        }
        return new Outcome(updated, updatedCost, !areMedoidsEqual(updated, currentMedoids));
    }

    // ---------------------------------------------------------------------------------------
    // Standalone entry point: the rounds above as ONE bounded iteration.
    // ---------------------------------------------------------------------------------------

    public static Result refine(
            PointSource source,
            EnvFactory envFactory,
            DenseVector[] initialMedoids,
            DenseVector[] candidatePool,
            DistanceMetric distanceMetric,
            int maxIterations) {

        if (maxIterations < 1) {
            return new Result(initialMedoids, Double.MAX_VALUE, 0);
        }

        int numClusters = initialMedoids.length;
        double[][] pool = extractCoordinates(candidatePool);
        double[][] medoids = extractCoordinates(initialMedoids);

        RefineState initialState = new RefineState();
        initialState.medoids = medoids;
        initialState.plan = planFor(medoids, pool, distanceMetric);
        initialState.cost = Double.MAX_VALUE;
        initialState.iterationCount = 0;

        RefineState finalState = MedoidIteration.execute(
            source, envFactory, "pamae-refine",
            initialState, RefineState.class, RefinePartial.class,
            new RefineLogic(pool, maxIterations, distanceMetric),
            NO_MERGE_STAGE);

        if (finalState == null) {
            return new Result(initialMedoids, Double.MAX_VALUE, 0);
        }

        DenseVector[] finalMedoids = new DenseVector[numClusters];
        for (int i = 0; i < numClusters; i++) {
            finalMedoids[i] = Points.wrap(finalState.medoids[i]);
        }
        return new Result(finalMedoids, finalState.cost, finalState.iterationCount);
    }

    /** Iteration state tracking current medoids and candidate grouping. */
    public static final class RefineState extends MedoidIteration.State {
        public double[][] medoids;
        public Plan plan;
        public double cost;
        public int iterationCount;

        public RefineState() {}
    }

    public static final class RefinePartial extends MedoidIteration.Partial {
        public double[] costs;

        public RefinePartial() {}
    }

    private static final class RefineLogic
            implements MedoidIteration.RoundLogic<RefineState, RefinePartial> {

        private final double[][] pool;
        private final int maxIterations;
        private final DistanceMetric distanceMetric;

        RefineLogic(double[][] pool, int maxIterations, DistanceMetric distanceMetric) {
            this.pool = pool;
            this.maxIterations = maxIterations;
            this.distanceMetric = distanceMetric;
        }

        @Override
        public RefinePartial computePartial(
                int round, RefineState state, Iterable<WeightedPoint> points, int subtaskId) {

            double[][] activeCandidates = materialize(state.plan, pool, state.medoids);
            RefinePartial partial = new RefinePartial();
            partial.costs = accumulate(
                points, state.medoids, activeCandidates, state.plan.clusterBounds, distanceMetric);
            return partial;
        }

        @Override
        public RefinePartial merge(RefinePartial left, RefinePartial right) {
            for (int i = 0; i < left.costs.length; i++) {
                left.costs[i] += right.costs[i];
            }
            return left;
        }

        @Override
        public MedoidIteration.Decision<RefineState> combine(
                int round, RefineState state, List<RefinePartial> partials) {

            double[] globalCosts = sumInOrder(partials);
            double[][] activeCandidates = materialize(state.plan, pool, state.medoids);
            Outcome outcome = chooseMedoids(
                activeCandidates, state.plan.clusterBounds, globalCosts, state.medoids);

            return decide(outcome, state, pool, distanceMetric, maxIterations);
        }

        private static double[] sumInOrder(List<RefinePartial> partials) {
            double[] total = new double[partials.get(0).costs.length];
            for (RefinePartial partial : partials) {
                for (int i = 0; i < total.length; i++) {
                    total[i] += partial.costs[i];
                }
            }
            return total;
        }
    }

    /** The stop/continue rule of a refinement round, shared with the merged {@code pamae} search. */
    public static MedoidIteration.Decision<RefineState> decide(
            Outcome outcome,
            RefineState state,
            double[][] pool,
            DistanceMetric distanceMetric,
            int maxIterations) {

        boolean improved = outcome.changed && outcome.cost < state.cost;
        RefineState next = new RefineState();
        next.medoids = improved ? outcome.medoids : state.medoids;
        next.cost = Math.min(state.cost, outcome.cost);
        next.iterationCount = state.iterationCount + 1;
        next.plan = improved ? planFor(next.medoids, pool, distanceMetric) : state.plan;

        return (!improved || next.iterationCount >= maxIterations)
            ? MedoidIteration.Decision.stop(next)
            : MedoidIteration.Decision.next(next);
    }

    // ---------------------------------------------------------------------------------------

    /** Finds the candidate index with the lowest cost for a given cluster. */
    private static int findBestCandidateForCluster(
            int clusterIndex,
            double[][] activeCandidates,
            int[] clusterBounds,
            double[] candidateCosts,
            double[][] currentMedoids) {

        int startIdx = clusterBounds[clusterIndex];
        int endIdx = clusterBounds[clusterIndex + 1];

        int bestIdx = -1;
        double minCost = 0.0;

        for (int i = startIdx; i < endIdx; i++) {
            if (activeCandidates[i] == currentMedoids[clusterIndex]
                    || Arrays.equals(activeCandidates[i], currentMedoids[clusterIndex])) {
                bestIdx = i;
                minCost = candidateCosts[i];
            }
        }
        for (int i = startIdx; i < endIdx; i++) {
            if (bestIdx < 0 || candidateCosts[i] < minCost) {
                bestIdx = i;
                minCost = candidateCosts[i];
            }
        }
        return bestIdx;
    }

    public static double[][] extractCoordinates(DenseVector[] vectors) {
        double[][] coords = new double[vectors.length][];
        for (int i = 0; i < coords.length; i++) {
            coords[i] = vectors[i].values;
        }
        return coords;
    }

    public static int findNearestCluster(double[] point, double[][] medoids, DistanceMetric distanceMetric) {
        int bestCluster = 0;
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < medoids.length; i++) {
            double dist = distanceMetric.compute(point, medoids[i]);
            if (dist < minDistance) {
                minDistance = dist;
                bestCluster = i;
            }
        }
        return bestCluster;
    }

    private static boolean areMedoidsEqual(double[][] first, double[][] second) {
        for (int i = 0; i < first.length; i++) {
            if (!Arrays.equals(first[i], second[i])) {
                return false;
            }
        }
        return true;
    }

    /**
     * Samples a uniform candidate pool across the entire dataset, as its own Flink job.
     *
     * <p>{@code pamae} does NOT call this — its pool is drawn inside the sampling round of its one
     * job, off the very pass that draws CLARA's samples, because a separate draw here means a
     * separate job and therefore a whole extra read of the source. It stays because it is the
     * pool primitive the refinement specs need to hold a pool fixed across runs.
     */
    public static DenseVector[] sampleCandidatePool(
            PointSource source, EnvFactory envFactory, long totalRowCount, int poolSize, long seed) {

        if (poolSize >= totalRowCount) {
            List<WeightedPoint> allPoints = Datasets.collectAll(source, envFactory);
            DenseVector[] pool = new DenseVector[allPoints.size()];
            for (int i = 0; i < pool.length; i++) {
                pool[i] = new DenseVector(allPoints.get(i).features.values.clone());
            }
            return pool;
        }

        double inclusionProb = DriverSample.calculateInclusionProbability(poolSize, totalRowCount);
        List<double[]> drawnSample = DriverSample.executeBernoulliSample(
            source, envFactory, inclusionProb, seed, "pamae-pool");

        double[][] selectedCoords = DriverSample.takeRandom(drawnSample, poolSize, seed);
        DenseVector[] pool = new DenseVector[selectedCoords.length];
        for (int i = 0; i < selectedCoords.length; i++) {
            pool[i] = Points.wrap(selectedCoords[i]);
        }
        return pool;
    }
}
