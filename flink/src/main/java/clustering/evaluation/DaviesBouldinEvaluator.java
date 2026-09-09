package clustering.evaluation;

import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import clustering.distance.EuclideanDistance;

/** {@link DaviesBouldinIndex} as a standalone evaluator: computes the moments itself.
 *
 *  The benchmark does NOT go through here — {@link clustering.benchmark.evaluation.EvaluationRunner}
 *  computes {@link ClusterMoments} once and feeds both indices from it, so a run that asks for both
 *  pays the two passes once. This entry point exists for the tests and for one-off use, mirroring
 *  the Spark {@code DaviesBouldinEvaluator}. */
public final class DaviesBouldinEvaluator implements ClusteringEvaluator {

    private final DistanceMetric distance;

    public DaviesBouldinEvaluator() {
        this(EuclideanDistance.INSTANCE);
    }

    public DaviesBouldinEvaluator(DistanceMetric distance) {
        this.distance = distance;
    }

    @Override
    public double evaluate(Model model, PointSource source, EnvFactory envs) {
        return DaviesBouldinIndex.of(ClusterMoments.compute(model, source, envs, distance), distance);
    }
}
