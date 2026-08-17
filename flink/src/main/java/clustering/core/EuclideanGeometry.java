package clustering.core;

import clustering.distance.DistanceMetric;
import clustering.distance.EuclideanDistance;

/** Plain Euclidean (Lloyd) geometry — the default, so old configs keep their meaning. */
public final class EuclideanGeometry implements Geometry {

    public static final EuclideanGeometry INSTANCE = new EuclideanGeometry();

    private EuclideanGeometry() {}

    @Override
    public String name() {
        return "euclidean";
    }

    @Override
    public PointSource prepare(PointSource source) {
        return source;
    }

    @Override
    public DistanceMetric fitDistance() {
        return EuclideanDistance.INSTANCE;
    }

    @Override
    public DistanceMetric modelDistance() {
        return EuclideanDistance.INSTANCE;
    }

    @Override
    public double[] project(double[] centroid) {
        return centroid;
    }
}
