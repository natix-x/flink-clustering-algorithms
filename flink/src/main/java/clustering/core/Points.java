package clustering.core;

import org.apache.flink.ml.linalg.DenseVector;
import java.util.ArrayList;
import java.util.List;

/** Unwrapping helpers for the boundary between the stream's record type ({@link WeightedPoint})
 *  and the raw {@code double[]} every driver-local phase and every distance loop works on.
 *
 *  {@link DenseVector#values} is the vector's OWN array, so unwrapping copies nothing — these
 *  helpers only rebuild the surrounding collection. */
public final class Points {

    private Points() {}

    public static double[] values(DenseVector vector) {
        return vector.values;
    }

    public static double[] values(WeightedPoint point) {
        return point.features.values;
    }

    /** Row-major view of a collected sample, for the driver-local algorithms. */
    public static double[][] toArray(List<WeightedPoint> points) {
        double[][] out = new double[points.size()][];
        for (int i = 0; i < out.length; i++) {
            out[i] = points.get(i).features.values;
        }
        return out;
    }

    /** The weights of a collected sample, index-aligned with {@link #toArray}. */
    public static double[] weightsOf(List<WeightedPoint> points) {
        double[] out = new double[points.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = points.get(i).weight;
        }
        return out;
    }


    public static List<double[]> toList(List<WeightedPoint> points) {
        List<double[]> out = new ArrayList<>(points.size());
        for (WeightedPoint p : points) {
            out.add(p.features.values);
        }
        return out;
    }

    public static DenseVector wrap(double[] values) {
        return new DenseVector(values);
    }

    /** Wraps raw rows as UNIT-WEIGHT stream records. Only the collection and the record wrappers
     *  are new — each vector shares the caller's array. */
    public static List<WeightedPoint> wrapAll(List<double[]> rows) {
        List<WeightedPoint> out = new ArrayList<>(rows.size());
        for (double[] row : rows) {
            out.add(WeightedPoint.withUnitWeight(row));
        }
        return out;
    }

    /** Wraps raw rows with explicit weights. */
    public static List<WeightedPoint> wrapAll(List<double[]> rows, double[] weights) {
        if (rows.size() != weights.length) {
            throw new IllegalArgumentException(
                "weights (" + weights.length + ") must match rows (" + rows.size() + ")");
        }
        List<WeightedPoint> out = new ArrayList<>(rows.size());
        for (int i = 0; i < weights.length; i++) {
            out.add(new WeightedPoint(new DenseVector(rows.get(i)), weights[i]));
        }
        return out;
    }

    /** Feature vectors as model-side records, for labelling raw rows. */
    public static List<DenseVector> vectorsOf(List<double[]> rows) {
        List<DenseVector> out = new ArrayList<>(rows.size());
        for (double[] row : rows) {
            out.add(new DenseVector(row));
        }
        return out;
    }
}
