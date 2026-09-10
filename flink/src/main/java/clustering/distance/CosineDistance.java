package clustering.distance;

/**
 * Computes the cosine distance (1 - cos(theta)) between two vectors.
 * Returns 1.0 (maximal dissimilarity) if either vector has a magnitude of zero.
 */
public final class CosineDistance implements DistanceMetric {

    public static final CosineDistance INSTANCE = new CosineDistance();

    private CosineDistance() {}

    @Override
    public double compute(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        double sumOfSquaresA = 0.0;
        double sumOfSquaresB = 0.0;

        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            sumOfSquaresA += vectorA[i] * vectorA[i];
            sumOfSquaresB += vectorB[i] * vectorB[i];
        }

        double magnitudeA = Math.sqrt(sumOfSquaresA);
        double magnitudeB = Math.sqrt(sumOfSquaresB);

        if (magnitudeA == 0.0 || magnitudeB == 0.0) {
            return 1.0;
        }

        return 1.0 - (dotProduct / (magnitudeA * magnitudeB));
    }
}
