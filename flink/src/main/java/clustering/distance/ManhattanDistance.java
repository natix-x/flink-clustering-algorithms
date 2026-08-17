package clustering.distance;

/** L1 distance. Singleton, mirrors the Scala {@code object ManhattanDistance}. */
public final class ManhattanDistance implements DistanceMetric {

    public static final ManhattanDistance INSTANCE = new ManhattanDistance();

    private ManhattanDistance() {}

    @Override
    public double compute(double[] x, double[] y) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += Math.abs(x[i] - y[i]);
        }
        return sum;
    }

    /** L1 sums are monotone, so the same early exit applies. */
    @Override
    public boolean withinRadius(double[] a, double[] b, double radius) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.abs(a[i] - b[i]);
            if (sum > radius) {
                return false;
            }
        }
        return sum <= radius;
    }
}
