package clustering.algorithms.kmedoids.components;

import clustering.distance.DistanceMetric;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents candidate medoid sets for the CLARA algorithm.
 * Collapses duplicate candidates across overlapping samples to ensure distinct distances
 * are computed exactly once per dataset point.
 */
public final class CandidateSets {

    private CandidateSets() {}

    /**
     * Collapses candidate coordinates onto their distinct values.
     */
    public static Deduplicated deduplicate(double[][] flatCandidates, int targetK) {
        int numSets = flatCandidates.length / targetK;
        Map<CoordinateKey, Integer> seen = new HashMap<>();
        int[] members = new int[flatCandidates.length];
        List<double[]> unique = new ArrayList<>();

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

    /**
     * Distinct candidates and their mapping back to the original sets and slots.
     */
    public static final class Deduplicated implements Serializable {

        public final double[][] unique;
        public final int[] members;
        public final int numSets;
        public final int targetK;

        Deduplicated(double[][] unique, int[] members, int numSets, int targetK) {
            this.unique = unique;
            this.members = members;
            this.numSets = numSets;
            this.targetK = targetK;
        }

        /**
         * Scratch buffer sized for distance computations in {@link #addPointCost}.
         */
        public double[] newDistanceBuffer() {
            return new double[unique.length];
        }

        /**
         * Adds a single point's distance contribution to every set's total cost.
         */
        public void addPointCost(
                double[] coordinates,
                double weight,
                DistanceMetric distanceMetric,
                double[] distanceBuffer,
                double[] costs
        ) {
            for (int u = 0; u < unique.length; u++) {
                distanceBuffer[u] = distanceMetric.compute(unique[u], coordinates);
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

    /**
     * Content-based key over a coordinate array for deduplication.
     */
    private static final class CoordinateKey {
        private final double[] coordinates;
        private final int hash;

        CoordinateKey(double[] coordinates) {
            this.coordinates = coordinates;
            this.hash = Arrays.hashCode(coordinates);
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof CoordinateKey &&
                   Arrays.equals(coordinates, ((CoordinateKey) other).coordinates);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }
}
