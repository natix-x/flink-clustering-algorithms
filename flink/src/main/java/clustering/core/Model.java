package clustering.core;

import org.apache.flink.ml.linalg.DenseVector;
import java.io.Serializable;
import java.util.List;

/** A fitted clustering model. Java mirror of the Spark repo's
 *  {@code clustering.core.Model}; points are {@link DenseVector} here, matching the record type
 *  the {@link PointSource} carries. Implementations unwrap {@code values} and work on the raw
 *  array, so the distance kernel is the same one both engines run. */
public interface Model extends Serializable {

    int predict(DenseVector features);

    /** Cluster label for each point (index-aligned with {@code data}). */
    default int[] labels(List<DenseVector> data) {
        int[] out = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            out[i] = predict(data.get(i));
        }
        return out;
    }
}