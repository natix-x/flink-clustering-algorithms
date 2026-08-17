package clustering.algorithms.kmeans;

import clustering.core.Clusterer;
import clustering.core.EnvFactory;
import clustering.core.Geometry;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/** Breathing k-means (Fritzke, 2020–2023) — the {@code refine: breathing} knob on slot 1, NOT
 *  a separate algorithm: it is an outer add/remove cycle around the same Lloyd update that
 *  plain {@link KMeans} runs, so the two are a controlled comparison.
 *
 *  Escapes Lloyd's local minima by temporarily growing to k+m centroids (inserting near the
 *  highest-error ones), re-converging, then shrinking back to k (dropping the lowest-utility
 *  ones, with a freeze rule to avoid removing a close pair together). Keeps the result if it
 *  improved, otherwise shrinks m and retries from the best solution so far; stops at m = 0.
 *
 *  <h3>Difference from the Spark implementation — one job, not one job per phase</h3>
 *  On Spark each phase is a separate {@code LloydKMeans.run} plus a separate statistics job,
 *  so a breathing run submits (cycles × phases × iterations) jobs and re-reads the cached
 *  DataFrame each time. Here the whole search — every Lloyd round of every phase of every
 *  cycle — is ONE Flink job: the phase machine below is a {@link CentroidIteration.RoundDriver}
 *  living in the parallelism-1 combiner, and the points stay cached per subtask for the entire
 *  run. The centroid set changes size between phases (k then k+m then k), which the shared
 *  iteration deliberately allows.
 *
 *  <h3>Cost of exactness: the MEASURE rounds</h3>
 *  Each round's statistics describe the centroid set the round was RUN AGAINST, so the SSE of
 *  a freshly converged set is not yet known when the phase ends. Rather than compare a
 *  one-step-stale SSE, the machine spends one extra distributed pass (a MEASURE round) on the
 *  converged set — the direct analogue of the extra Spark job {@code computeTotalError} runs,
 *  and it doubles as the source of the error/utility statistics the next breath needs.
 *
 *  @param m0        initial number of centroids added/removed per cycle (paper default 5)
 *  @param maxIter   maximum Lloyd rounds per phase
 *  @param maxCycles safety bound on breathing cycles; the paper's own bound is m reaching 0 */
public class BreathingKMeans implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(BreathingKMeans.class);

    private final int k;
    private final int m0;
    private final int maxIter;
    private final double eps;
    private final long seed;
    private final Geometry geometry;
    private final int maxCycles;

    public BreathingKMeans(int k, int m0, int maxIter, double eps, long seed,
                           Geometry geometry, int maxCycles) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1, got " + k);
        }
        if (m0 < 1) {
            throw new IllegalArgumentException("m0 must be >= 1, got " + m0);
        }
        this.k = k;
        this.m0 = m0;
        this.maxIter = maxIter;
        this.eps = eps;
        this.seed = seed;
        this.geometry = geometry;
        this.maxCycles = maxCycles;
    }

    @Override
    public KMeansModel fit(PointSource source, EnvFactory envs, int parallelism) {
        PointSource prepared = geometry.prepare(source);
        double[][] init = CentroidIteration.sampleInitialCentroids(prepared, envs, k, seed);
        double[][] best = CentroidIteration.run(
            prepared, envs, init, geometry.fitDistance(),
            new BreathingDriver(k, m0, maxIter, eps, seed, geometry, maxCycles), "breathing-kmeans-fit");
        return new KMeansModel(best, geometry.modelDistance());
    }

    /** The breathing phase machine. Runs in the parallelism-1 combiner and keeps its own
     *  mutable search state across rounds (one instance, one task, all rounds). */
    static final class BreathingDriver implements CentroidIteration.RoundDriver {

        /** Insertion offset scale, epsilon in the paper. */
        private static final double INSERTION_SCALE = 0.01;

        private enum Phase {
            /** Lloyd rounds on the k initial centroids. */
            LLOYD_INITIAL,
            /** One pass to score the converged initial set, then breathe in. */
            MEASURE_INITIAL,
            /** Lloyd rounds on the k+m expanded set. */
            LLOYD_EXPANDED,
            /** One pass for the expanded set's utilities, then breathe out. */
            MEASURE_EXPANDED,
            /** Lloyd rounds on the set shrunk back to k. */
            LLOYD_REDUCED,
            /** One pass to score the shrunk set against the best so far. */
            MEASURE_REDUCED,
            /** One pass to re-read the best set's error/utility after a failed cycle. */
            MEASURE_BEST
        }

        private final int k;
        private final int maxIter;
        private final double eps;
        private final Geometry geometry;
        private final int maxCycles;
        private final Random random;

        private Phase phase = Phase.LLOYD_INITIAL;
        private int lloydRounds;
        private int currentM;
        private int cycles;
        private double[][] bestCentroids;
        private double bestError = Double.MAX_VALUE;

        BreathingDriver(int k, int m0, int maxIter, double eps, long seed,
                        Geometry geometry, int maxCycles) {
            this.k = k;
            this.maxIter = maxIter;
            this.eps = eps;
            this.geometry = geometry;
            this.maxCycles = maxCycles;
            this.currentM = m0;
            this.random = new Random(seed);
        }

        @Override
        public CentroidIteration.Decision nextRound(int epoch, CentroidIteration.RoundStats stats) {
            switch (phase) {
                case LLOYD_INITIAL:
                    return lloyd(stats, Phase.MEASURE_INITIAL);
                case LLOYD_EXPANDED:
                    return lloyd(stats, Phase.MEASURE_EXPANDED);
                case LLOYD_REDUCED:
                    return lloyd(stats, Phase.MEASURE_REDUCED);
                case MEASURE_INITIAL:
                    bestCentroids = stats.centroids;
                    bestError = stats.totalError();
                    logger.info("breathing: k={} m={} initial SSE={}", k, currentM, bestError);
                    return breatheIn(stats);
                case MEASURE_EXPANDED:
                    return breatheOut(stats);
                case MEASURE_REDUCED:
                    return closeCycle(stats);
                case MEASURE_BEST:
                    return breatheIn(stats);
                default:
                    throw new IllegalStateException("unreachable phase: " + phase);
            }
        }

        /** One Lloyd round; on convergence (or the {@code maxIter} bound) the next round is
         *  {@code measurePhase}, which scores the converged set exactly. */
        private CentroidIteration.Decision lloyd(CentroidIteration.RoundStats stats, Phase measurePhase) {
            double[][] next = stats.means(geometry);
            double movement = CentroidIteration.maxMovement(stats.centroids, next, geometry.fitDistance());
            lloydRounds++;
            if (movement < eps || lloydRounds >= maxIter) {
                phase = measurePhase;
                lloydRounds = 0;
            }
            return CentroidIteration.Decision.cont(next);
        }

        /** Breathe in: insert one new centroid next to each of the {@code currentM}
         *  highest-error centroids. At most one insertion per existing centroid, so a breath is
         *  capped at |C| — with k &lt; m fewer than m centroids are actually inserted. */
        private CentroidIteration.Decision breatheIn(CentroidIteration.RoundStats stats) {
            double[][] centroids = stats.centroids;
            long mass = stats.totalCount();
            double rmse = mass > 0L ? Math.sqrt(stats.totalError() / mass) : 0.0;

            Integer[] byError = indices(centroids.length);
            // Descending error, ties by ascending index — the Spark tie rule.
            java.util.Arrays.sort(byError, Comparator
                .<Integer>comparingDouble(i -> -stats.errors[i])
                .thenComparingInt(i -> i));

            int insertions = Math.min(currentM, centroids.length);
            double[][] expanded = new double[centroids.length + insertions][];
            System.arraycopy(centroids, 0, expanded, 0, centroids.length);
            for (int i = 0; i < insertions; i++) {
                expanded[centroids.length + i] = offset(centroids[byError[i]], rmse);
            }

            phase = Phase.LLOYD_EXPANDED;
            lloydRounds = 0;
            logger.debug("breathing: breathe in m={} |C|={} -> {}", currentM, centroids.length, expanded.length);
            return CentroidIteration.Decision.cont(expanded);
        }

        /** Breathe out: shrink back to k. Removes exactly as many centroids as were inserted —
         *  the invariant is "shrink back to k", not "remove m", so a capped breath cannot empty
         *  the centroid set. */
        private CentroidIteration.Decision breatheOut(CentroidIteration.RoundStats stats) {
            double[][] expanded = stats.centroids;
            int removalCount = expanded.length - k;
            double[][] reduced;
            if (removalCount <= 0) {
                reduced = expanded;
            } else {
                Set<Integer> removed = selectForRemoval(expanded, stats.utilities, removalCount);
                List<double[]> kept = new ArrayList<>(k);
                for (int i = 0; i < expanded.length; i++) {
                    if (!removed.contains(i)) {
                        kept.add(expanded[i]);
                    }
                }
                reduced = kept.toArray(new double[0][]);
            }
            phase = Phase.LLOYD_REDUCED;
            lloydRounds = 0;
            logger.debug("breathing: breathe out |C|={} -> {}", expanded.length, reduced.length);
            return CentroidIteration.Decision.cont(reduced);
        }

        /** Compare the shrunk set with the best so far, then either breathe in again or stop.
         *  A failed cycle costs one unit of breath and the search restarts from the best set —
         *  which needs its error/utility statistics again, hence the {@code MEASURE_BEST} pass. */
        private CentroidIteration.Decision closeCycle(CentroidIteration.RoundStats stats) {
            double candidateError = stats.totalError();
            boolean improved = candidateError < bestError;
            if (improved) {
                bestCentroids = stats.centroids;
                bestError = candidateError;
            } else {
                currentM--;
            }
            cycles++;
            logger.info("breathing: cycle={} m={} SSE={} best={}", cycles, currentM, candidateError, bestError);

            if (currentM <= 0 || cycles >= maxCycles) {
                if (cycles >= maxCycles) {
                    logger.warn("breathing: stopped at the maxCycles={} bound with m={}", maxCycles, currentM);
                }
                return CentroidIteration.Decision.stop(bestCentroids);
            }
            if (improved) {
                // These statistics already describe the new best set — breathe in directly.
                return breatheIn(stats);
            }
            phase = Phase.MEASURE_BEST;
            return CentroidIteration.Decision.cont(bestCentroids);
        }

        /** {@code c + INSERTION_SCALE · RMSE · u}, u uniform in the unit hypercube centred at
         *  the origin. Projected onto the geometry, so on the unit sphere the inserted centroid
         *  stays a unit vector. */
        private double[] offset(double[] centroid, double rmse) {
            double[] perturbed = new double[centroid.length];
            for (int i = 0; i < centroid.length; i++) {
                perturbed[i] = centroid[i] + INSERTION_SCALE * rmse * (random.nextDouble() - 0.5);
            }
            return geometry.project(perturbed);
        }

        /** Indices of the {@code removalCount} centroids to delete: lowest utility first,
         *  freezing the nearest neighbour of every centroid picked, as long as
         *  {@code |frozen| + removalCount < |C|}. */
        private Set<Integer> selectForRemoval(double[][] centroids, double[] utilities, int removalCount) {
            Integer[] byUtility = indices(centroids.length);
            java.util.Arrays.sort(byUtility, Comparator
                .<Integer>comparingDouble(i -> utilities[i])
                .thenComparingInt(i -> i));

            DistanceMetric metric = geometry.fitDistance();
            Set<Integer> removed = new HashSet<>();
            Set<Integer> frozen = new HashSet<>();
            for (int idx : byUtility) {
                if (removed.size() < removalCount && !frozen.contains(idx)) {
                    removed.add(idx);
                    if (frozen.size() + removalCount < centroids.length) {
                        int neighbour = nearestNeighbour(centroids, idx, metric);
                        if (neighbour >= 0) {
                            frozen.add(neighbour);
                        }
                    }
                }
            }
            // A pathological freeze pattern could leave fewer than removalCount removals; fall
            // back to the plain utility ranking for the remainder so the set always shrinks to k.
            for (int idx : byUtility) {
                if (removed.size() >= removalCount) {
                    break;
                }
                removed.add(idx);
            }
            return removed;
        }

        private static int nearestNeighbour(double[][] centroids, int idx, DistanceMetric metric) {
            int nearest = -1;
            double nearestDistance = Double.MAX_VALUE;
            for (int i = 0; i < centroids.length; i++) {
                if (i != idx) {
                    double d = metric.compute(centroids[idx], centroids[i]);
                    if (d < nearestDistance) {
                        nearestDistance = d;
                        nearest = i;
                    }
                }
            }
            return nearest;
        }

        private static Integer[] indices(int length) {
            Integer[] out = new Integer[length];
            for (int i = 0; i < length; i++) {
                out[i] = i;
            }
            return out;
        }
    }
}
