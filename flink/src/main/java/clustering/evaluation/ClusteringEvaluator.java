package clustering.evaluation;

import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;

import java.io.Serializable;

/**
 * Evaluates the internal quality of a fitted clustering model over a dataset.
 */
public interface ClusteringEvaluator extends Serializable {

    /**
     * Computes the evaluation metric for the given model.
     */
    double evaluate(Model model, PointSource source, EnvFactory envFactory);
}
