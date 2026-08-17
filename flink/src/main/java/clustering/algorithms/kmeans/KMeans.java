package clustering.algorithms.kmeans;

import clustering.core.Clusterer;
import clustering.core.EnvFactory;
import clustering.core.EuclideanGeometry;
import clustering.core.Geometry;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Lloyd's k-means or spherical k-means (Dhillon &amp; Modha 2001) depending on the
 *  {@code geometry} knob — mirror of the Spark {@code clustering.algorithms.kmeans.KMeans}.
 *
 *  The whole training is ONE Flink job on the Flink ML bounded-iteration framework
 *  (FLIP-176): see {@link CentroidIteration} for the job shape. The algorithm itself is ours
 *  (init, assignment, mean update, empty-cluster handling, convergence); only the iteration
 *  runtime is Flink ML's. */
public class KMeans implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(KMeans.class);

    private final int k;
    private final int maxIter;
    private final double eps;
    private final long seed;
    private final Geometry geometry;

    public KMeans(int k, int maxIter, double eps, long seed, Geometry geometry) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1, got " + k);
        }
        this.k = k;
        this.maxIter = maxIter;
        this.eps = eps;
        this.seed = seed;
        this.geometry = geometry;
    }

    /** Euclidean geometry, the default — kept so callers that only pick a metric-free k-means
     *  do not have to name the geometry. */
    public KMeans(int k, int maxIter, double eps, long seed) {
        this(k, maxIter, eps, seed, EuclideanGeometry.INSTANCE);
    }

    @Override
    public KMeansModel fit(PointSource source, EnvFactory envs, int parallelism) {
        PointSource prepared = geometry.prepare(source);
        double[][] init = CentroidIteration.sampleInitialCentroids(prepared, envs, k, seed);
        double[][] centroids = CentroidIteration.run(
            prepared, envs, init, geometry.fitDistance(),
            new LloydDriver(geometry, maxIter, eps), "kmeans-fit");
        logger.debug("kmeans: k={} geometry={} maxIter={} eps={}", k, geometry.name(), maxIter, eps);
        return new KMeansModel(centroids, geometry.modelDistance());
    }

    /** Plain Lloyd: every round is one mean update; stop on convergence (largest centroid
     *  movement below {@code eps}) or after {@code maxIter} rounds.
     *
     *  Package-private because {@link BreathingKMeans} runs the very same update — that is
     *  what makes {@code refine: breathing} a knob on one algorithm rather than a second
     *  algorithm. */
    static final class LloydDriver implements CentroidIteration.RoundDriver {

        private final Geometry geometry;
        private final int maxIter;
        private final double eps;

        LloydDriver(Geometry geometry, int maxIter, double eps) {
            this.geometry = geometry;
            this.maxIter = maxIter;
            this.eps = eps;
        }

        @Override
        public CentroidIteration.Decision nextRound(int epoch, CentroidIteration.RoundStats stats) {
            double[][] next = stats.means(geometry);
            double movement = CentroidIteration.maxMovement(stats.centroids, next, geometry.fitDistance());
            boolean stop = movement < eps || epoch >= maxIter - 1;
            if (stop) {
                logger.debug("kmeans: stop at round={} (maxMove={}, eps={}, maxIter={})",
                    epoch, movement, eps, maxIter);
            }
            return new CentroidIteration.Decision(next, stop);
        }
    }
}
