package clustering.distance;

import clustering.core.SphericalGeometry;

/** Cosine distance specialised to <b>unit-norm</b> vectors: {@code 1 - <a, b>}.
 *
 *  Identical to {@link CosineDistance} whenever both arguments are L2-normalised, but it
 *  skips the two {@code sqrt(sum x^2)} passes — exactly the per-iteration saving spherical
 *  k-means is supposed to deliver at high dimensionality.
 *
 *  Deliberately NOT registered in {@code DistanceRegistry}: it is only correct on normalised
 *  data, so it may not be selected from a config. It is chosen internally by
 *  {@link clustering.core.SphericalGeometry}, which guarantees the normalisation. */
public final class UnitSphereDistance implements DistanceMetric {

    public static final UnitSphereDistance INSTANCE = new UnitSphereDistance();

    private UnitSphereDistance() {}

    @Override
    public double compute(double[] a, double[] b) {
        double dot = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return 1.0 - dot;
    }
}
