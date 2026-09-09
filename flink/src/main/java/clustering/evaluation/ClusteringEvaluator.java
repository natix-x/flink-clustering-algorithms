package clustering.evaluation;

import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;
import java.io.Serializable;

/** One internal quality index of a fitted model, over the FULL dataset.
 *
 *  Java mirror of the Spark repo's {@code clustering.evaluation.ClusteringEvaluator}. The seam
 *  differs the way the two engines' seams differ everywhere else: Spark passes the labelled
 *  {@code DataFrame}, Flink passes the {@link PointSource} plus an {@link EnvFactory}, because a
 *  Flink env is bound to one job and every action needs a fresh one.
 *
 *  {@link SilhouetteEvaluator} implements this too — its full-data path — but the benchmark calls
 *  its {@code measure} overload instead, which subsamples and reports what it scored. */
public interface ClusteringEvaluator extends Serializable {

    double evaluate(Model model, PointSource source, EnvFactory envs);
}
