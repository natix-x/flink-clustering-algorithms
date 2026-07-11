package clustering.algorithms.kmedoids;

import clustering.core.Clusterer;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;

import java.util.List;

/** FastPAM — optimized variant of Partitioning Around Medoids. Java mirror of the Spark
 *  {@code FastPAM}.
 *
 *  FastPAM reduces the SWAP-phase complexity from O(k²*n²) to O(k*n²) by caching the
 *  nearest (d1) and second-nearest (d2) medoid distances for every point. This avoids
 *  re-evaluating the whole cluster configuration on every swap test.
 *
 *  Like PAM it is driver-local (O(n²) distance matrix), so suitable only for datasets
 *  that fit in driver memory. */
public class FastPAM implements Clusterer {

    private final int k;
    private final int maxIter;
    private final DistanceMetric distance;

    public FastPAM(int k, int maxIter, DistanceMetric distance) {
        this.k = k;
        this.maxIter = maxIter;
        this.distance = distance;
    }

    @Override
    public Model fit(PointSource source, EnvFactory envs, int parallelism) {
        // Collect the full dataset to the driver (parallelism 1, deterministic order).
        // This implementation assumes the dataset fits on a single machine.
        List<double[]> collected = Datasets.collectAll(source, envs);
        return fitLocal(collected.toArray(new double[0][]));
    }

    /** Local entry point — no Flink overhead. All computation happens on the driver. */
    public KMedoidsModel fitLocal(double[][] points) {
        int n = points.length;
        if (n < k) {
            throw new IllegalArgumentException(
                "Dataset too small: n=" + n + " points but k=" + k + " medoids requested.");
        }

        // Step 1: precompute all pairwise distances into a flat 1D array.
        double[] dist = precomputeDistances(points, n);

        // Step 2: BUILD phase — greedy heuristic initialization.
        int[] medoids = buildPhase(dist, n);
        int iter = 0;
        boolean improved = true;

        // Step 3: pre-allocate state arrays to avoid allocations inside the loop.
        boolean[] isMedoid = new boolean[n];
        double[] d1 = new double[n];
        double[] d2 = new double[n];
        int[] nearestMedoidIdx = new int[n];

        // Step 4: iterative optimization using FastPAM swap logic.
        while (improved && iter < maxIter) {
            improved = swapPhaseFast(dist, n, medoids, isMedoid, d1, d2, nearestMedoidIdx);
            iter++;
        }

        double[][] medoidPoints = new double[k][];
        for (int i = 0; i < k; i++) {
            medoidPoints[i] = points[medoids[i]];
        }
        return new KMedoidsModel(medoidPoints, distance);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /** Pairwise distances in a flat 1D array (dist[i*n + j]) for spatial locality. */
    private double[] precomputeDistances(double[][] points, int n) {
        double[] dist = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double d = distance.compute(points[i], points[j]);
                dist[i * n + j] = d;
                dist[j * n + i] = d;
            }
        }
        return dist;
    }

    /** BUILD phase: greedily selects initial medoids to minimise global distance. */
    private int[] buildPhase(double[] dist, int n) {
        int[] selected = new int[k];
        int selCount = 0;

        // Step A: first medoid — minimises total distance to all other points.
        int bestFirst = 0;
        double bestSum = Double.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) {
                s += dist[i * n + j];
            }
            if (s < bestSum) {
                bestSum = s;
                bestFirst = i;
            }
        }
        selected[0] = bestFirst;
        selCount++;

        // dBest[j] = minimum distance from point j to any currently chosen medoid.
        double[] dBest = new double[n];
        for (int j = 0; j < n; j++) {
            dBest[j] = dist[bestFirst * n + j];
        }

        // Step B: successively select remaining k-1 medoids.
        while (selCount < k) {
            int bestH = -1;
            double bestGain = Double.NEGATIVE_INFINITY;

            for (int h = 0; h < n; h++) {
                boolean alreadySelected = false;
                for (int c = 0; c < selCount && !alreadySelected; c++) {
                    if (selected[c] == h) {
                        alreadySelected = true;
                    }
                }
                if (!alreadySelected) {
                    double gain = 0.0;
                    for (int j = 0; j < n; j++) {
                        double g = dBest[j] - dist[h * n + j];
                        if (g > 0.0) {
                            gain += g;
                        }
                    }
                    if (gain > bestGain) {
                        bestGain = gain;
                        bestH = h;
                    }
                }
            }
            selected[selCount] = bestH;
            selCount++;

            for (int j = 0; j < n; j++) {
                double d = dist[bestH * n + j];
                if (d < dBest[j]) {
                    dBest[j] = d;
                }
            }
        }

        return selected;
    }

    /** FastPAM SWAP phase: evaluates potential swaps in O(k*n²). Instead of recomputing
     *  configuration costs from scratch, it derives the exact cost change (delta) from
     *  whether a point loses its closest medoid or finds a closer alternative. Mutates
     *  {@code medoids} in place and returns whether an improving swap was applied. */
    private boolean swapPhaseFast(
            double[] dist,
            int n,
            int[] medoids,
            boolean[] isMedoid,
            double[] d1,
            double[] d2,
            int[] nearestMedoidIdx) {

        // Reset and populate the fast medoid-lookup boolean array.
        for (int i = 0; i < n; i++) {
            isMedoid[i] = false;
        }
        for (int mi = 0; mi < k; mi++) {
            isMedoid[medoids[mi]] = true;
        }

        // Precompute d1 (closest), d2 (second closest) and nearestMedoidIdx per point.
        for (int j = 0; j < n; j++) {
            double min1 = Double.MAX_VALUE;
            double min2 = Double.MAX_VALUE;
            int m1Idx = -1;

            for (int m = 0; m < k; m++) {
                double d = dist[j * n + medoids[m]];
                if (d < min1) {
                    min2 = min1;
                    min1 = d;
                    m1Idx = m; // index within the medoids array [0, k-1]
                } else if (d < min2) {
                    min2 = d;
                }
            }
            d1[j] = min1;
            d2[j] = min2;
            nearestMedoidIdx[j] = m1Idx;
        }

        int bestSwapMi = -1;
        int bestSwapH = -1;
        double bestDelta = 0.0; // look for the most negative delta (maximum savings)

        // Evaluate every non-medoid point 'h' as a replacement candidate.
        for (int h = 0; h < n; h++) {
            if (!isMedoid[h]) {
                for (int mi = 0; mi < k; mi++) {
                    double currentDelta = 0.0;
                    for (int j = 0; j < n; j++) {
                        double dh = dist[j * n + h];

                        if (nearestMedoidIdx[j] == mi) {
                            // Case 1: point j loses its primary medoid; reassign to the
                            // new candidate 'h' or its second-nearest medoid d2[j].
                            currentDelta += (Math.min(dh, d2[j]) - d1[j]);
                        } else {
                            // Case 2: point j keeps its primary medoid, unless 'h' is closer.
                            if (dh < d1[j]) {
                                currentDelta += (dh - d1[j]);
                            }
                        }
                    }

                    if (currentDelta < bestDelta) {
                        bestDelta = currentDelta;
                        bestSwapMi = mi;
                        bestSwapH = h;
                    }
                }
            }
        }

        if (bestDelta < 0.0) {
            medoids[bestSwapMi] = bestSwapH;
            return true;
        }
        return false; // local optimum reached
    }
}