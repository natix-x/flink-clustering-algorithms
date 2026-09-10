package clustering.algorithms.dbscan;

import clustering.algorithms.dbscan.components.EpsilonGraphComponents;
import clustering.algorithms.dbscan.components.EpsilonNeighbourCounter;
import clustering.algorithms.dbscan.components.UniformSelection;
import clustering.core.Clusterer;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * DBSCAN++ implementation (Jang & Jiang, ICML 2019).
 * Approximates core points using a sub-sampled candidate set, reducing the
 * graph construction time complexity from O(n²) to O(n·m).
 *
 * @param coreSampleFraction fraction of data to sample as candidates (s in (0, 1])
 * @param strictEpsilonCheck true for classic DBSCAN noise handling; false to assign all points
 * @param chunkSize          batch size for distributed neighborhood counting
 */
public class DBSCANpp implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(DBSCANpp.class);

    private final double epsilon;
    private final int minPoints;
    private final double coreSampleFraction;
    private final UniformSelection candidateSelector;
    private final boolean strictEpsilonCheck;
    private final int chunkSize;
    private final DistanceMetric distanceMetric;
    private final long seed;

    public DBSCANpp(double epsilon, int minPoints, double coreSampleFraction,
                    UniformSelection candidateSelector, boolean strictEpsilonCheck,
                    int chunkSize, DistanceMetric distanceMetric, long seed) {
        validateArgs(epsilon, minPoints, coreSampleFraction, chunkSize);
        this.epsilon = epsilon;
        this.minPoints = minPoints;
        this.coreSampleFraction = coreSampleFraction;
        this.candidateSelector = candidateSelector;
        this.strictEpsilonCheck = strictEpsilonCheck;
        this.chunkSize = chunkSize;
        this.distanceMetric = distanceMetric;
        this.seed = seed;
    }

    private static void validateArgs(double epsilon, int minPoints, double coreSampleFraction, int chunkSize) {
        if (!(epsilon > 0.0) || Double.isNaN(epsilon)) {
            throw new IllegalArgumentException("epsilon must be > 0, got " + epsilon);
        }
        if (minPoints < 1) {
            throw new IllegalArgumentException("minPoints must be >= 1, got " + minPoints);
        }
        if (!(coreSampleFraction > 0.0) || coreSampleFraction > 1.0) {
            throw new IllegalArgumentException("coreSampleFraction must be in (0, 1], got " + coreSampleFraction);
        }
        if (chunkSize < 1) {
            throw new IllegalArgumentException("chunkSize must be >= 1, got " + chunkSize);
        }
    }

    @Override
    public CoreLabelModel fit(PointSource source, EnvFactory envFactory, int parallelism) {
        long totalPoints = Datasets.count(source, envFactory);

        int targetCandidateCount = (int) Math.max(1L,
            Math.min(totalPoints, (long) Math.ceil(coreSampleFraction * totalPoints)));

        double[][] sampledCandidates = candidateSelector.selectCandidates(
            source, envFactory, totalPoints, targetCandidateCount, seed, distanceMetric);

        logger.info("dbscanpp: totalPoints={} m={} sampling={} epsilon={} minPoints={} s={}",
            totalPoints, sampledCandidates.length, candidateSelector.strategyName(),
            epsilon, minPoints, coreSampleFraction);

        double[] densityCounts = EpsilonNeighbourCounter.computeNeighbourhoodDensities(
            source, envFactory, sampledCandidates, epsilon, distanceMetric, chunkSize);

        List<double[]> validCores = new ArrayList<>();
        double totalCoreDegree = 0.0;

        for (int i = 0; i < sampledCandidates.length; i++) {
            if (densityCounts[i] >= minPoints) {
                validCores.add(sampledCandidates[i]);
                totalCoreDegree += densityCounts[i];
            }
        }

        double[][] corePoints = validCores.toArray(new double[0][]);
        logger.info("dbscanpp: {} of {} candidates are core points", corePoints.length, sampledCandidates.length);

        if (corePoints.length == 0) {
            logger.warn("dbscanpp: no core point found (epsilon={}, minPoints={}) — all points are noise",
                epsilon, minPoints);
            return new CoreLabelModel(new double[0][], new int[0], epsilon, distanceMetric, strictEpsilonCheck);
        }

        double estimatedCoreDegree = totalPoints == 0L ? 0.0 : totalCoreDegree / totalPoints;

        int[] clusterLabels = EpsilonGraphComponents.compute(
            corePoints, epsilon, distanceMetric, envFactory, parallelism, estimatedCoreDegree);

        logger.info("dbscanpp: {} clusters", (int) Arrays.stream(clusterLabels).distinct().count());

        return new CoreLabelModel(corePoints, clusterLabels, epsilon, distanceMetric, strictEpsilonCheck);
    }
}
