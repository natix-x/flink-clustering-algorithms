package clustering.algorithms.kmeans;

import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Clusterer;
import clustering.core.EnvFactory;
import clustering.core.EuclideanGeometry;
import clustering.core.Geometry;
import clustering.core.PointSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of Lloyd's k-means or spherical k-means clustering.
 * The training is executed as a single Flink job using bounded iterations.
 */
public class KMeans implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(KMeans.class);

    private final int k;
    private final int maxIterations;
    private final double tolerance;
    private final long seed;
    private final Geometry geometry;

    public KMeans(int k, int maxIterations, double tolerance, long seed, Geometry geometry) {
        this.k = k;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.seed = seed;
        this.geometry = geometry;
    }

    /** Initializes KMeans with default Euclidean geometry. */
    public KMeans(int k, int maxIterations, double tolerance, long seed) {
        this(k, maxIterations, tolerance, seed, EuclideanGeometry.INSTANCE);
    }

    @Override
    public KMeansModel fit(PointSource source, EnvFactory envFactory, int parallelism) {
        PointSource preparedSource = geometry.prepare(source);
        DenseVector[] initialCentroids = CentroidIteration.sampleInitialCentroids(preparedSource, envFactory, k, seed);

        DenseVector[] finalCentroids = CentroidIteration.execute(
            preparedSource, envFactory, initialCentroids, geometry.fitDistance(),
            new LloydDriver(geometry, maxIterations, tolerance), "kmeans-fit");

        logger.debug("kmeans: k={} geometry={} maxIterations={} tolerance={}", k, geometry.name(), maxIterations, tolerance);
        return new KMeansModel(finalCentroids, geometry.modelDistance());
    }

    /**
     * Driver for standard Lloyd iterations.
     * Stops upon convergence or when the maximum number of iterations is reached.
     */
    static final class LloydDriver implements CentroidIteration.IterationDriver {

        private final Geometry geometry;
        private final int maxIterations;
        private final double tolerance;

        LloydDriver(Geometry geometry, int maxIterations, double tolerance) {
            this.geometry = geometry;
            this.maxIterations = maxIterations;
            this.tolerance = tolerance;
        }

        @Override
        public CentroidIteration.DriverDecision computeNextRound(int epoch, CentroidIteration.IterationStats stats) {
            DenseVector[] updatedCentroids = stats.computeMeans(geometry);
            double maxMovement = CentroidIteration.calculateMaxMovement(stats.centroids, updatedCentroids, geometry.fitDistance());
            boolean shouldStop = maxMovement < tolerance || epoch >= maxIterations - 1;

            if (shouldStop) {
                logger.debug("kmeans: stop at round={} (maxMovement={}, tolerance={}, maxIterations={})",
                    epoch, maxMovement, tolerance, maxIterations);
            }

            return new CentroidIteration.DriverDecision(updatedCentroids, shouldStop);
        }
    }
}
