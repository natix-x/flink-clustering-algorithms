package clustering.distance;

/** L2 distance. Singleton, mirrors the Scala {@code object EuclideanDistance}. */
public final class EuclideanDistance implements DistanceMetric {

    public static final EuclideanDistance INSTANCE = new EuclideanDistance();

    private EuclideanDistance() {}

    @Override
    public double compute(double[] x, double[] y) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            double d = x[i] - y[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }
}