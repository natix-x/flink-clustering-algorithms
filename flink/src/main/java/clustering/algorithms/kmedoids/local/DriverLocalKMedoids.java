package clustering.algorithms.kmedoids.local;

import clustering.algorithms.kmedoids.KMedoidsModel;
import clustering.core.WeightedPoint;
import clustering.core.Clusterer;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.core.Points;
import clustering.core.Weights;
import clustering.distance.DistanceMetric;

/**
 * Interface for driver-local k-medoids solvers that retain the O(n²) distance matrix in memory.
 */
public interface DriverLocalKMedoids extends Clusterer {

    KMedoidsModel fitLocal(double[][] points, double[] weights);

    default KMedoidsModel fitLocal(double[][] points) {
        return fitLocal(points, Weights.unit(points.length));
    }

    @Override
    default Model fit(PointSource source, EnvFactory envFactory, int parallelism) {
        java.util.List<WeightedPoint> rows = Datasets.collectAll(source, envFactory);
        return fitLocal(Points.toArray(rows), Points.weightsOf(rows));
    }

    static DriverLocalKMedoids fromName(
            String name, int k, int maxIterations, DistanceMetric distanceMetric, long seed) {
        switch (name.toLowerCase()) {
            case "fastpam":
                return new FastPAM(k, maxIterations, distanceMetric);
            case "fasterpam":
                return new FasterPAM(k, maxIterations, distanceMetric, seed);
            default:
                throw new IllegalArgumentException(
                    "Unknown inner k-medoids solver: '" + name + "'. Supported: fastpam, fasterpam");
        }
    }

    static KMedoidsModel modelOf(double[][] points, int[] medoids, DistanceMetric distanceMetric) {
        org.apache.flink.ml.linalg.DenseVector[] medoidVectors =
            new org.apache.flink.ml.linalg.DenseVector[medoids.length];
        for (int slot = 0; slot < medoids.length; slot++) {
            medoidVectors[slot] = Points.wrap(points[medoids[slot]]);
        }
        return new KMedoidsModel(medoidVectors, distanceMetric);
    }
}
