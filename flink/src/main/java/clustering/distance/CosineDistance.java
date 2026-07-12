package clustering.distance;

/** Cosine distance = {@code 1 - cos(theta)}. Singleton, mirrors the Scala
 *  {@code object CosineDistance}. Returns {@code 1.0} when either vector is zero
 *  (undefined cosine -> maximal dissimilarity), matching the Spark implementation. */
public final class CosineDistance implements DistanceMetric {

    public static final CosineDistance INSTANCE = new CosineDistance();

    private CosineDistance() {}

    @Override
    public double compute(double[] x, double[] y) {
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < x.length; i++) {
            dot += x[i] * y[i];
            normA += x[i] * x[i];
            normB += y[i] * y[i];
        }
        normA = Math.sqrt(normA);
        normB = Math.sqrt(normB);
        if (normA == 0.0 || normB == 0.0) {
            return 1.0;
        }
        return 1.0 - (dot / (normA * normB));
    }
}
