package clustering.distance;

import org.apache.flink.ml.linalg.DenseVector;
import java.io.Serializable;

/** A distance between two feature vectors. Java mirror of the Spark repo's
 *  {@code clustering.distance.DistanceMetric}.
 *
 *  <h3>Two entry points, one kernel</h3>
 *  {@link #compute(DenseVector, DenseVector)} is the seam every algorithm uses — the same
 *  signature the Spark side has, over that engine's own vector type. It delegates to the raw
 *  {@code double[]} kernel through {@link DenseVector#values}, which is the vector's OWN array,
 *  so the delegation copies nothing and allocates nothing.
 *
 *  The array form stays public for callers whose data is flat by construction (the driver-local
 *  distance matrices, the ε-scans): wrapping those in vectors just to unwrap them again inside
 *  the loop would allocate per comparison — the exact anti-pattern that cost 5x when the ε-scan
 *  unpacked coordinates per comparison instead of per row. */
public interface DistanceMetric extends Serializable {

    double compute(double[] a, double[] b);

    /** Vector-typed seam, mirroring the Spark {@code compute(Vector, Vector)}. */
    default double compute(DenseVector a, DenseVector b) {
        return compute(a.values, b.values);
    }

    /** {@code d(a, b) <= radius}, for callers that only need the PREDICATE and never the distance.
     *
     *  Exists because a metric can usually answer it far more cheaply than it can produce the
     *  distance: it skips the {@code sqrt} and, more importantly, can exit the coordinate loop the
     *  moment the partial sum passes the radius. In an ε-scan almost no pair is within ε, so most
     *  comparisons end after a coordinate or two regardless of dimensionality — which is what makes
     *  {@code dbscanpp}'s two scans (n·m in step 2, m²/2 in step 3) affordable at 256–1024 dims.
     *  Measured on the Spark side of this thesis, where the same change was made to the same two
     *  scans: 5x on 2·10^6 points × 6 000 candidates.
     *
     *  Must agree with {@link #compute} exactly, including at the boundary — {@code <=} is what
     *  makes a point ON the ε-sphere a neighbour. Squaring is exact for the comparison: for
     *  non-negative x and r, {@code x <= r} iff {@code x² <= r²}. NaN coordinates make both forms
     *  false, as they do in {@code compute}. */
    default boolean withinRadius(double[] a, double[] b, double radius) {
        return compute(a, b) <= radius;
    }

    /** {@code d(a, b)} when it is {@code <= bound}, else {@link Double#POSITIVE_INFINITY} — the
     *  NEAREST-PROTOTYPE counterpart of {@link #withinRadius}, for a scan that keeps a running
     *  minimum (or, tracking a top-2, a running SECOND minimum).
     *
     *  Same early exit, with a bound that SHRINKS as the scan finds better candidates: a caller
     *  passes {@code min(eps, bestSoFar)} or the current second-nearest distance, so every hit
     *  tightens the test for the rest of the scan. Mirrors the Spark repo's
     *  {@code DistanceMetric.distanceUpTo} exactly — same semantics, same name.
     *
     *  Why returning infinity rather than the true distance is not a loss of information: the
     *  caller is looking for the minimum (or the top-2), and anything above the bound cannot be
     *  either slot, so the caller never needed its exact value.
     *
     *  Must agree with {@link #compute} whenever it returns a finite value, boundary included
     *  ({@code d == bound} is a hit). NaN coordinates yield infinity, matching {@link #withinRadius}'s
     *  {@code false}. */
    default double distanceUpTo(double[] a, double[] b, double bound) {
        double d = compute(a, b);
        return d <= bound ? d : Double.POSITIVE_INFINITY;
    }
}
