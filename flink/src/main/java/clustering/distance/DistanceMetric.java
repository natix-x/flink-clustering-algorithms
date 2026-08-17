package clustering.distance;

import java.io.Serializable;

/** A distance between two feature vectors. Java mirror of the Spark repo's
 *  {@code clustering.distance.DistanceMetric}. */
public interface DistanceMetric extends Serializable {

    double compute(double[] a, double[] b);

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
}
