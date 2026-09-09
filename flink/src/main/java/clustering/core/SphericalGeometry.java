package clustering.core;

import clustering.distance.CosineDistance;
import clustering.distance.DistanceMetric;
import clustering.distance.UnitSphereDistance;
import org.apache.flink.ml.linalg.DenseVector;

/** Spherical geometry (Dhillon &amp; Modha 2001) — the unit hypersphere.
 *
 *  {@link #prepare} composes an L2-normalising map onto the point source, so the
 *  normalisation runs inside whatever job the algorithm builds (one extra map per record,
 *  no extra pass and no extra materialisation). */
public final class SphericalGeometry implements Geometry {

    public static final SphericalGeometry INSTANCE = new SphericalGeometry();

    private SphericalGeometry() {}

    @Override
    public String name() {
        return "spherical";
    }

    @Override
    public PointSource prepare(PointSource source) {
        // Only the FEATURES are normalised; the weight is a multiplicity, not a coordinate, so it
        // rides through untouched.
        return env -> source.create(env)
            .map(p -> new WeightedPoint(
                new DenseVector(Geometry.l2Normalize(p.features.values)), p.weight))
            .returns(WeightedPointTypeInfo.INSTANCE)
            .name("l2-normalize");
    }

    /** On normalised data cosine collapses to {@code 1 - dot}. */
    @Override
    public DistanceMetric fitDistance() {
        return UnitSphereDistance.INSTANCE;
    }

    /** Cosine, not {@code 1 - dot}: the model labels raw, un-normalised points and cosine
     *  is scale-invariant, so the labels are the same as on normalised data. */
    @Override
    public DistanceMetric modelDistance() {
        return CosineDistance.INSTANCE;
    }

    @Override
    public DenseVector project(DenseVector centroid) {
        return new DenseVector(Geometry.l2Normalize(centroid.values));
    }
}
