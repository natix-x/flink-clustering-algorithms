package clustering.distance;

/**
 * Computes the cosine distance specialized for unit-norm (L2-normalized) vectors.
 * Assumes inputs are already normalized to safely skip magnitude calculations.
 */
public final class UnitSphereDistance implements DistanceMetric {

    public static final UnitSphereDistance INSTANCE = new UnitSphereDistance();

    private UnitSphereDistance() {}

    @Override
    public double compute(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;

        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
        }

        return 1.0 - dotProduct;
    }
}
