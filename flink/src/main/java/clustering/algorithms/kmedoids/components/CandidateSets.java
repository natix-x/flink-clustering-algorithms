package clustering.algorithms.kmedoids.components;

import clustering.distance.DistanceMetric;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The {@code numSamples} candidate medoid sets CLARA has to score against the entire dataset,
 * with duplicate candidates collapsed.
 *
 * <p>Scoring is the pass that touches every row: for each point it needs
 * {@code min} over each set's k medoids, i.e. {@code numSamples · k} distance computations per
 * point written naively. But the sets are the solutions of k-medoids runs on overlapping samples of
 * one dataset, so they agree on part of their medoids far more often than not — and every shared
 * medoid was being measured against every point once per set that contains it. Collapsing them
 * computes each distinct distance ONCE and then takes the per-set minima off an index table, which
 * turns the inner loop from {@code numSamples · k} distance computations (each O(d)) into
 * {@code |unique|} of them plus {@code numSamples · k} array reads. The result is identical, not
 * approximate — the same numbers in a different order of evaluation, and the minimum does not care.
 *
 * <p>Worth it at the dimensionalities this benchmark runs: at d = 1024 a distance is three orders of
 * magnitude more work than an array lookup, so the saving is essentially the duplicate fraction.
 */
public final class CandidateSets {

    private CandidateSets() {}

    /** Collapses {@code numSets · k} candidate coordinates onto their distinct values. */
    public static Deduplicated deduplicate(double[][] flatCandidates, int targetK) {
        int numSets = flatCandidates.length / targetK;
        Map<CoordinateKey, Integer> seen = new HashMap<>();
        int[] members = new int[flatCandidates.length];
        List<double[]> unique = new java.util.ArrayList<>();

        for (int i = 0; i < flatCandidates.length; i++) {
            CoordinateKey key = new CoordinateKey(flatCandidates[i]);
            Integer index = seen.get(key);
            if (index == null) {
                index = unique.size();
                unique.add(flatCandidates[i]);
                seen.put(key, index);
            }
            members[i] = index;
        }
        return new Deduplicated(unique.toArray(new double[0][]), members, numSets, targetK);
    }

    /** Distinct candidates plus, for every (set, slot), which distinct candidate it is. */
    public static final class Deduplicated implements Serializable {

        public final double[][] unique;
        /** {@code numSets · k} indices into {@link #unique}, set-major. */
        public final int[] members;
        public final int numSets;
        public final int targetK;

        Deduplicated(double[][] unique, int[] members, int numSets, int targetK) {
            this.unique = unique;
            this.members = members;
            this.numSets = numSets;
            this.targetK = targetK;
        }

        /** Scratch buffer of the right size for {@link #addPointCost}. */
        public double[] newDistanceBuffer() {
            return new double[unique.length];
        }

        /** Adds one point's contribution to every set's cost. */
        public void addPointCost(
                double[] coordinates, double weight, DistanceMetric metric,
                double[] distanceBuffer, double[] costs) {

            for (int u = 0; u < unique.length; u++) {
                distanceBuffer[u] = metric.compute(unique[u], coordinates);
            }
            for (int set = 0; set < numSets; set++) {
                int base = set * targetK;
                double nearest = Double.MAX_VALUE;
                for (int slot = 0; slot < targetK; slot++) {
                    double distance = distanceBuffer[members[base + slot]];
                    if (distance < nearest) {
                        nearest = distance;
                    }
                }
                costs[set] += weight * nearest;
            }
        }
    }

    /** Content-based key over a coordinate array. */
    private static final class CoordinateKey {
        private final double[] coordinates;
        private final int hash;

        CoordinateKey(double[] coordinates) {
            this.coordinates = coordinates;
            this.hash = Arrays.hashCode(coordinates);
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof CoordinateKey && Arrays.equals(coordinates, ((CoordinateKey) other).coordinates);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }
}
