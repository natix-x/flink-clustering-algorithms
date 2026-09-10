package clustering.core;

import org.apache.flink.ml.linalg.DenseVector;

import java.io.Serializable;
import java.util.List;

/**
 * A fitted clustering model.
 */
public interface Model extends Serializable {

    /**
     * Returns the predicted cluster index for the given features.
     */
    int predict(DenseVector features);

    /**
     * Predicts cluster labels for a batch of points.
     */
    default int[] predictAll(List<DenseVector> points) {
        int[] predictions = new int[points.size()];
        for (int i = 0; i < points.size(); i++) {
            predictions[i] = predict(points.get(i));
        }
        return predictions;
    }
}
