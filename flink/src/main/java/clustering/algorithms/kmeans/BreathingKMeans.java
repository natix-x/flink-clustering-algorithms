package clustering.algorithms.kmeans;

import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Clusterer;
import clustering.core.EnvFactory;
import clustering.core.Geometry;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Breathing k-means implementation.
 * Escapes local minima by cyclically adding (breathing in) and removing (breathing out) centroids,
 * wrapping standard Lloyd iterations in a single Flink job.
 */
public class BreathingKMeans implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(BreathingKMeans.class);

    private final int targetK;
    private final int initialBreathSize;
    private final int maxIterations;
    private final double tolerance;
    private final long seed;
    private final Geometry geometry;
    private final int maxCycles;

    public BreathingKMeans(int targetK, int initialBreathSize, int maxIterations, double tolerance, long seed,
                           Geometry geometry, int maxCycles) {
        if (targetK < 1) {
            throw new IllegalArgumentException("Target k must be >= 1, got " + targetK);
        }
        if (initialBreathSize < 1) {
            throw new IllegalArgumentException("Initial breath size must be >= 1, got " + initialBreathSize);
        }
        this.targetK = targetK;
        this.initialBreathSize = initialBreathSize;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.seed = seed;
        this.geometry = geometry;
        this.maxCycles = maxCycles;
    }

    @Override
    public KMeansModel fit(PointSource source, EnvFactory envFactory, int parallelism) {
        PointSource preparedSource = geometry.prepare(source);
        DenseVector[] initialCentroids = CentroidIteration.sampleInitialCentroids(preparedSource, envFactory, targetK, seed);

        DenseVector[] bestCentroids = CentroidIteration.execute(
            preparedSource, envFactory, initialCentroids, geometry.fitDistance(),
            new BreathingDriver(targetK, initialBreathSize, maxIterations, tolerance, seed, geometry, maxCycles), "breathing-kmeans-fit");

        return new KMeansModel(bestCentroids, geometry.modelDistance());
    }

    /** State machine managing the breathing phases across iteration rounds. */
    static final class BreathingDriver implements CentroidIteration.IterationDriver {

        private static final double INSERTION_SCALE = 0.01;

        private enum Phase {
            LLOYD_INITIAL,
            MEASURE_INITIAL,
            LLOYD_EXPANDED,
            MEASURE_EXPANDED,
            LLOYD_REDUCED,
            MEASURE_REDUCED,
            MEASURE_BEST
        }

        private final int targetK;
        private final int maxIterations;
        private final double tolerance;
        private final Geometry geometry;
        private final int maxCycles;
        private final Random randomGenerator;

        private Phase currentPhase = Phase.LLOYD_INITIAL;
        private int currentIterationCount;
        private int currentBreathSize;
        private int completedCycles;
        private DenseVector[] bestCentroids;
        private double minError = Double.MAX_VALUE;

        BreathingDriver(int targetK, int initialBreathSize, int maxIterations, double tolerance, long seed,
                        Geometry geometry, int maxCycles) {
            this.targetK = targetK;
            this.maxIterations = maxIterations;
            this.tolerance = tolerance;
            this.geometry = geometry;
            this.maxCycles = maxCycles;
            this.currentBreathSize = initialBreathSize;
            this.randomGenerator = new Random(seed);
        }

        @Override
        public CentroidIteration.DriverDecision computeNextRound(int epoch, CentroidIteration.IterationStats stats) {
            switch (currentPhase) {
                case LLOYD_INITIAL:
                    return executeLloydRound(stats, Phase.MEASURE_INITIAL);
                case LLOYD_EXPANDED:
                    return executeLloydRound(stats, Phase.MEASURE_EXPANDED);
                case LLOYD_REDUCED:
                    return executeLloydRound(stats, Phase.MEASURE_REDUCED);
                case MEASURE_INITIAL:
                    bestCentroids = stats.centroids;
                    minError = stats.calculateTotalError();
                    logger.info("breathing: k={} breathSize={} initial SSE={}", targetK, currentBreathSize, minError);
                    return expandCentroids(stats);
                case MEASURE_EXPANDED:
                    return reduceCentroids(stats);
                case MEASURE_REDUCED:
                    return evaluateCycle(stats);
                case MEASURE_BEST:
                    return expandCentroids(stats);
                default:
                    throw new IllegalStateException("Unreachable phase: " + currentPhase);
            }
        }

        private CentroidIteration.DriverDecision executeLloydRound(CentroidIteration.IterationStats stats, Phase nextPhase) {
            DenseVector[] updatedCentroids = stats.computeMeans(geometry);
            double maxMovement = CentroidIteration.calculateMaxMovement(stats.centroids, updatedCentroids, geometry.fitDistance());

            currentIterationCount++;
            if (maxMovement < tolerance || currentIterationCount >= maxIterations) {
                currentPhase = nextPhase;
                currentIterationCount = 0;
            }
            return CentroidIteration.DriverDecision.continueIteration(updatedCentroids);
        }

        /** Inserts new centroids near the highest-error existing centroids. */
        private CentroidIteration.DriverDecision expandCentroids(CentroidIteration.IterationStats stats) {
            DenseVector[] currentCentroids = stats.centroids;
            long totalMass = stats.calculateTotalCount();
            double rmse = totalMass > 0L ? Math.sqrt(stats.calculateTotalError() / totalMass) : 0.0;

            Integer[] sortedIndices = generateIndices(currentCentroids.length);
            Arrays.sort(sortedIndices, Comparator.<Integer>comparingDouble(i -> -stats.clusterErrors[i]).thenComparingInt(i -> i));

            int insertCount = Math.min(currentBreathSize, currentCentroids.length);
            DenseVector[] expandedCentroids = new DenseVector[currentCentroids.length + insertCount];
            System.arraycopy(currentCentroids, 0, expandedCentroids, 0, currentCentroids.length);

            for (int i = 0; i < insertCount; i++) {
                expandedCentroids[currentCentroids.length + i] = applyJitter(currentCentroids[sortedIndices[i]], rmse);
            }

            currentPhase = Phase.LLOYD_EXPANDED;
            currentIterationCount = 0;
            logger.debug("breathing: expanding |C|={} -> {}", currentCentroids.length, expandedCentroids.length);
            return CentroidIteration.DriverDecision.continueIteration(expandedCentroids);
        }

        /** Removes lowest-utility centroids to shrink the set back to targetK. */
        private CentroidIteration.DriverDecision reduceCentroids(CentroidIteration.IterationStats stats) {
            DenseVector[] expandedCentroids = stats.centroids;
            int removeCount = expandedCentroids.length - targetK;
            DenseVector[] reducedCentroids;

            if (removeCount <= 0) {
                reducedCentroids = expandedCentroids;
            } else {
                Set<Integer> indicesToRemove = identifyCentroidsToRemove(expandedCentroids, stats.clusterUtilities, removeCount);
                List<DenseVector> keptCentroids = new ArrayList<>(targetK);
                for (int i = 0; i < expandedCentroids.length; i++) {
                    if (!indicesToRemove.contains(i)) {
                        keptCentroids.add(expandedCentroids[i]);
                    }
                }
                reducedCentroids = keptCentroids.toArray(new DenseVector[0]);
            }

            currentPhase = Phase.LLOYD_REDUCED;
            currentIterationCount = 0;
            logger.debug("breathing: reducing |C|={} -> {}", expandedCentroids.length, reducedCentroids.length);
            return CentroidIteration.DriverDecision.continueIteration(reducedCentroids);
        }

        /** Evaluates if the cycle improved the clustering error, adjusts breath size accordingly. */
        private CentroidIteration.DriverDecision evaluateCycle(CentroidIteration.IterationStats stats) {
            double currentError = stats.calculateTotalError();
            boolean isImproved = currentError < minError;

            if (isImproved) {
                bestCentroids = stats.centroids;
                minError = currentError;
            } else {
                currentBreathSize--;
            }
            completedCycles++;
            logger.info("breathing: cycle={} breathSize={} SSE={} best={}", completedCycles, currentBreathSize, currentError, minError);

            if (currentBreathSize <= 0 || completedCycles >= maxCycles) {
                if (completedCycles >= maxCycles) {
                    logger.warn("breathing: stopped at maxCycles={} bound", maxCycles);
                }
                return CentroidIteration.DriverDecision.stopIteration(bestCentroids);
            }

            if (isImproved) {
                return expandCentroids(stats);
            }

            currentPhase = Phase.MEASURE_BEST;
            return CentroidIteration.DriverDecision.continueIteration(bestCentroids);
        }

        private DenseVector applyJitter(DenseVector centroid, double rmse) {
            double[] coordinates = centroid.values;
            double[] perturbedCoordinates = new double[coordinates.length];
            for (int i = 0; i < coordinates.length; i++) {
                perturbedCoordinates[i] = coordinates[i] + INSERTION_SCALE * rmse * (randomGenerator.nextDouble() - 0.5);
            }
            return geometry.project(new DenseVector(perturbedCoordinates));
        }

        private Set<Integer> identifyCentroidsToRemove(DenseVector[] centroids, double[] utilities, int targetRemoveCount) {
            Integer[] sortedIndices = generateIndices(centroids.length);
            Arrays.sort(sortedIndices, Comparator.<Integer>comparingDouble(i -> utilities[i]).thenComparingInt(i -> i));

            DistanceMetric metric = geometry.fitDistance();
            Set<Integer> removedIndices = new HashSet<>();
            Set<Integer> frozenIndices = new HashSet<>();

            for (int idx : sortedIndices) {
                if (removedIndices.size() < targetRemoveCount && !frozenIndices.contains(idx)) {
                    removedIndices.add(idx);
                    if (frozenIndices.size() + targetRemoveCount < centroids.length) {
                        int nearestNeighbor = findNearestNeighbor(centroids, idx, metric);
                        if (nearestNeighbor >= 0) {
                            frozenIndices.add(nearestNeighbor);
                        }
                    }
                }
            }

            for (int idx : sortedIndices) {
                if (removedIndices.size() >= targetRemoveCount) {
                    break;
                }
                removedIndices.add(idx);
            }
            return removedIndices;
        }

        private static int findNearestNeighbor(DenseVector[] centroids, int targetIdx, DistanceMetric metric) {
            int nearestIdx = -1;
            double minDistance = Double.MAX_VALUE;
            for (int i = 0; i < centroids.length; i++) {
                if (i != targetIdx) {
                    double dist = metric.compute(centroids[targetIdx], centroids[i]);
                    if (dist < minDistance) {
                        minDistance = dist;
                        nearestIdx = i;
                    }
                }
            }
            return nearestIdx;
        }

        private static Integer[] generateIndices(int length) {
            Integer[] indices = new Integer[length];
            for (int i = 0; i < length; i++) {
                indices[i] = i;
            }
            return indices;
        }
    }
}
