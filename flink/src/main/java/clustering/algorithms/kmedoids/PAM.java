package clustering.algorithms.kmedoids;

import clustering.core.Clusterer;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.Points;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;

import java.util.List;

/** Partitioning Around Medoids (PAM) — classic k-medoids clustering. Java mirror of
 *  the Spark {@code PAM}.
 *
 *  PAM is more robust to noise and outliers than KMeans because it uses actual data
 *  points (medoids) as cluster centres instead of arbitrary centroids.
 *
 *  Scalability note: PAM is O(k*(n-k)²) per iteration and needs O(n²) memory for the
 *  distance matrix, so it is only feasible for small datasets (n ≤ ~10 000). For large
 *  datasets use {@link CLARA}, which runs PAM on small random samples.
 *
 *  Two entry points:
 *    {@link #fit} — collects the whole point stream to the driver and runs PAM locally.
 *    {@link #fitLocal} — runs PAM on an already-collected array; used by CLARA to avoid
 *      the collect round-trip. */
public class PAM implements Clusterer {

    private final int k;
    private final int maxIter;
    private final DistanceMetric distance;

    public PAM(int k, int maxIter, DistanceMetric distance) {
        this.k = k;
        this.maxIter = maxIter;
        this.distance = distance;
    }

    /** Collects the full dataset to the driver (parallelism 1, deterministic order) and
     *  delegates to {@link #fitLocal}. Only suitable for datasets that fit in driver
     *  memory. */
    @Override
    public Model fit(PointSource source, EnvFactory envs, int parallelism) {
        return fitLocal(Points.toArray(Datasets.collectAll(source, envs)));
    }

    /** Local entry point used by CLARA — no Flink overhead. All computation happens on
     *  the driver using flat arrays. */
    public KMedoidsModel fitLocal(double[][] points) {
        int n = points.length;
        if (n < k) {
            throw new IllegalArgumentException(
                "Dataset too small: n=" + n + " points but k=" + k + " medoids requested.");
        }

        // Step 1: precompute all pairwise distances into a flat 1D array.
        double[] dist = precomputeDistances(points, n);

        // Step 2: BUILD phase — greedy initial medoid selection.
        int[] medoids = buildPhase(dist, n);
        int iter = 0;
        boolean improved = true;

        // Step 3: SWAP phase — iteratively improve until convergence or maxIter.
        while (improved && iter < maxIter) {
            improved = swapPhaseStandard(dist, n, medoids);
            iter++;
        }

        double[][] medoidPoints = new double[k][];
        for (int i = 0; i < k; i++) {
            medoidPoints[i] = points[medoids[i]];
        }
        return new KMedoidsModel(medoidPoints, distance);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /** Pairwise distance matrix in a flat 1D array (dist[i*n + j]). Flat layout improves
     *  CPU cache locality and avoids 2D-array pointer chasing. Only the upper triangle is
     *  computed; symmetry fills the rest. */
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

    /** BUILD phase: greedy selection of k initial medoids.
     *  1. First medoid: the point minimising total distance to all others.
     *  2. Remaining k-1: each chosen to maximise incremental cost reduction. */
    private int[] buildPhase(double[] dist, int n) {
        int[] selected = new int[k];
        int selCount = 0;

        // Step A: first medoid — closest to the dataset's centre of mass.
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

        // dBest[j] = distance from point j to its nearest selected medoid.
        double[] dBest = new double[n];
        for (int j = 0; j < n; j++) {
            dBest[j] = dist[bestFirst * n + j];
        }

        // Step B: greedily add remaining k-1 medoids.
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

    /** Standard SWAP phase: O(k*(n-k)²) exhaustive search. Tests every (medoid,
     *  non-medoid) swap and applies the one with the greatest cost reduction. Mutates
     *  {@code medoids} in place and returns whether an improving swap was applied. */
    private boolean swapPhaseStandard(double[] dist, int n, int[] medoids) {
        int bestSwapMi = -1;
        int bestSwapH = -1;
        double bestDelta = 0.0;

        int[] candidateMedoids = medoids.clone();

        for (int mi = 0; mi < k; mi++) {
            int oldMedoid = medoids[mi];

            for (int h = 0; h < n; h++) {
                boolean isAlreadyMedoid = false;
                for (int mIdx = 0; mIdx < k; mIdx++) {
                    if (medoids[mIdx] == h) {
                        isAlreadyMedoid = true;
                    }
                }

                if (!isAlreadyMedoid) {
                    candidateMedoids[mi] = h;

                    double currentDelta = 0.0;
                    for (int j = 0; j < n; j++) {
                        double oldMin = Double.MAX_VALUE;
                        for (int m = 0; m < k; m++) {
                            double d = dist[j * n + medoids[m]];
                            if (d < oldMin) {
                                oldMin = d;
                            }
                        }

                        double newMin = Double.MAX_VALUE;
                        for (int m = 0; m < k; m++) {
                            double d = dist[j * n + candidateMedoids[m]];
                            if (d < newMin) {
                                newMin = d;
                            }
                        }

                        currentDelta += (newMin - oldMin);
                    }

                    if (currentDelta < bestDelta) {
                        bestDelta = currentDelta;
                        bestSwapMi = mi;
                        bestSwapH = h;
                    }

                    candidateMedoids[mi] = oldMedoid;
                }
            }
        }

        if (bestDelta < 0.0) {
            medoids[bestSwapMi] = bestSwapH;
            return true;
        }
        return false;
    }
}