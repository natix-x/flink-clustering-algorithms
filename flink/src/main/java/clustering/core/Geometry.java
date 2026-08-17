package clustering.core;

import clustering.distance.DistanceMetric;
import org.apache.flink.ml.linalg.DenseVector;

import java.io.Serializable;

/** The space a centroid-based algorithm optimises in — a knob, not an algorithm
 *  (euclidean = Lloyd k-means, spherical = Dhillon &amp; Modha 2001). Java mirror of the
 *  Spark {@code clustering.core.Geometry}; the metric is fixed by the geometry, not a
 *  free config param.
 *
 *  Four hooks:
 *    - {@link #prepare} — one transformation of the fit input. On Spark this rewrites the
 *      {@code features} column of a DataFrame; here it WRAPS the {@link PointSource}, so the
 *      normalisation becomes a map operator inside every job the algorithm runs (nothing is
 *      materialised twice). Idempotent, so an outer algorithm may prepare and let an inner
 *      one prepare again;
 *    - {@link #fitDistance} — the metric used inside the iteration loop, on prepared data;
 *    - {@link #modelDistance} — the metric the fitted model uses to label RAW data, since
 *      evaluation labels the untransformed source (cosine is scale-invariant, so labels agree);
 *    - {@link #project} — the projection applied to each updated centroid. */
public interface Geometry extends Serializable {

    /** Registry name, as it appears in a run config. */
    String name();

    PointSource prepare(PointSource source);

    DistanceMetric fitDistance();

    DistanceMetric modelDistance();

    DenseVector project(DenseVector centroid);

    /** L2-normalises {@code v} into a NEW array; a zero vector is returned unchanged (both
     *  cosine variants already treat it as maximally distant). */
    static double[] l2Normalize(double[] v) {
        double sumOfSquares = 0.0;
        for (double x : v) {
            sumOfSquares += x * x;
        }
        double norm = Math.sqrt(sumOfSquares);
        if (norm == 0.0) {
            return v;
        }
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i] / norm;
        }
        return out;
    }
}
