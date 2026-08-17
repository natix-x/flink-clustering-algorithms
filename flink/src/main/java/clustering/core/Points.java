package clustering.core;

import org.apache.flink.ml.linalg.DenseVector;

import java.util.ArrayList;
import java.util.List;

/** Unwrapping helpers for the boundary between the stream's record type ({@link DenseVector})
 *  and the raw {@code double[]} every driver-local phase and every distance loop works on.
 *
 *  {@link DenseVector#values} is the vector's OWN array, so unwrapping copies nothing — these
 *  helpers only rebuild the surrounding collection. */
public final class Points {

    private Points() {}

    public static double[] values(DenseVector vector) {
        return vector.values;
    }

    /** Row-major view of a collected sample, for the driver-local algorithms. */
    public static double[][] toArray(List<DenseVector> vectors) {
        double[][] out = new double[vectors.size()][];
        for (int i = 0; i < out.length; i++) {
            out[i] = vectors.get(i).values;
        }
        return out;
    }

    public static List<double[]> toList(List<DenseVector> vectors) {
        List<double[]> out = new ArrayList<>(vectors.size());
        for (DenseVector v : vectors) {
            out.add(v.values);
        }
        return out;
    }

    public static DenseVector wrap(double[] values) {
        return new DenseVector(values);
    }

    /** Wraps raw rows as stream records. Only the collection is new — each vector shares the
     *  caller's array. */
    public static List<DenseVector> wrapAll(List<double[]> rows) {
        List<DenseVector> out = new ArrayList<>(rows.size());
        for (double[] row : rows) {
            out.add(new DenseVector(row));
        }
        return out;
    }
}