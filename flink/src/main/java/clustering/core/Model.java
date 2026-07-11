package clustering.core;

import java.io.Serializable;
import java.util.List;

/** A fitted clustering model. Java mirror of the Spark repo's
 *  {@code clustering.core.Model}; points are plain {@code double[]} here. */
public interface Model extends Serializable {

    int predict(double[] features);

    /** Cluster label for each point (index-aligned with {@code data}). */
    default int[] labels(List<double[]> data) {
        int[] out = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            out[i] = predict(data.get(i));
        }
        return out;
    }
}