package clustering.evaluation;

import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import clustering.distance.EuclideanDistance;

/** {@link CalinskiHarabaszIndex} as a standalone evaluator: computes the moments itself.
 *
 *  As with {@link DaviesBouldinEvaluator}, the benchmark shares one {@link ClusterMoments} between
 *  the two indices instead of calling this — see
 *  {@link clustering.benchmark.evaluation.EvaluationRunner}. */
public final class CalinskiHarabaszEvaluator implements ClusteringEvaluator {

    private final DistanceMetric distance;

    public CalinskiHarabaszEvaluator() {
        this(EuclideanDistance.INSTANCE);
    }

    public CalinskiHarabaszEvaluator(DistanceMetric distance) {
        this.distance = distance;
    }

    @Override
    public double evaluate(Model model, PointSource source, EnvFactory envs) {
        return CalinskiHarabaszIndex.of(ClusterMoments.compute(model, source, envs, distance), distance);
    }
}
