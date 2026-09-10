package clustering.evaluation;

import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import clustering.distance.EuclideanDistance;

/**
 * Evaluates a clustering model using the Davies-Bouldin Index.
 */
public final class DaviesBouldinEvaluator implements ClusteringEvaluator {

    private final DistanceMetric distanceMetric;

    public DaviesBouldinEvaluator() {
        this(EuclideanDistance.INSTANCE);
    }

    public DaviesBouldinEvaluator(DistanceMetric distanceMetric) {
        this.distanceMetric = distanceMetric;
    }

    @Override
    public double evaluate(Model model, PointSource source, EnvFactory envFactory) {
        ClusterMoments moments = ClusterMoments.compute(model, source, envFactory, distanceMetric);
        return DaviesBouldinIndex.compute(moments, distanceMetric);
    }
}
