package clustering.distance;


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

    /** Squared distances, with an early exit: once the partial sum passes r² the remaining
     *  coordinates cannot bring it back. */
    @Override
    public boolean withinRadius(double[] a, double[] b, double radius) {
        double limit = radius * radius;
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
            if (sum > limit) {
                return false;
            }
        }
        return sum <= limit;
    }

    /** Squared-space early exit, {@code sqrt} paid only for a hit. */
    @Override
    public double distanceUpTo(double[] a, double[] b, double bound) {
        double limit = bound * bound;
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
            if (sum > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }
        return sum <= limit ? Math.sqrt(sum) : Double.POSITIVE_INFINITY;
    }
}
