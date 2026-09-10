package clustering.evaluation;

import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import clustering.distance.EuclideanDistance;

/**
 * Evaluates a clustering model using the Calinski-Harabasz Index.
 */
public final class CalinskiHarabaszEvaluator implements ClusteringEvaluator {

    private final DistanceMetric distanceMetric;

    public CalinskiHarabaszEvaluator() {
        this(EuclideanDistance.INSTANCE);
    }

    public CalinskiHarabaszEvaluator(DistanceMetric distanceMetric) {
        this.distanceMetric = distanceMetric;
    }

    @Override
    public double evaluate(Model model, PointSource source, EnvFactory envFactory) {
        ClusterMoments moments = ClusterMoments.compute(model, source, envFactory, distanceMetric);
        return CalinskiHarabaszIndex.compute(moments, distanceMetric);
    }
}
