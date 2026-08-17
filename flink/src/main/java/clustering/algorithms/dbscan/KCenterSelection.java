package clustering.algorithms.dbscan;

import clustering.core.EnvFactory;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;

import java.util.stream.IntStream;

/** Greedy K-center (farthest-first traversal): repeatedly add the point farthest from those
 *  chosen so far — a 2-approximation of the minimax radius, so coverage beats uniform.
 *
 *  <b>Documented deviation</b> (identical to the Spark side). The paper traverses the full
 *  dataset, which would need m sequential distributed rounds. Here it runs on the driver over a
 *  uniform POOL of {@code poolFactor * m} points: one Flink job, then O(m * |pool| * d) across
 *  the driver's cores. The 2-approximation then holds against the pool, not the full data — say
 *  so in the thesis. {@code poolFactor} is a second accuracy-vs-cost knob: at 1 the strategy IS
 *  uniform, and where its advantage saturates is a cheap empirical result. */
public final class KCenterSelection implements CandidateSelectionStrategy {

    /** Below this, splitting the scan across cores costs more than the scan it saves. */
    private static final int MinParallelPoolSize = 4096;

    private final int poolFactor;

    public KCenterSelection(int poolFactor) {
        if (poolFactor < 1) {
            throw new IllegalArgumentException("poolFactor must be >= 1, got " + poolFactor);
        }
        this.poolFactor = poolFactor;
    }

    @Override
    public String strategyName() {
        return "kcenter";
    }

    @Override
    public double[][] selectCandidates(PointSource source, EnvFactory envs, long totalRowCount,
                                       int targetCandidateCount, long seed, DistanceMetric distanceMetric) {
        int poolSize = (int) Math.min(totalRowCount, (long) targetCandidateCount * poolFactor);
        double[][] pool = UniformSelection.INSTANCE
            .selectCandidates(source, envs, totalRowCount, poolSize, seed, distanceMetric);
        if (targetCandidateCount >= pool.length) {
            return pool;
        }

        int[] selected = new int[targetCandidateCount];
        boolean[] taken = new boolean[pool.length];
        // The paper starts arbitrarily; fixing the start at index 0 keeps it deterministic
        // per seed (the pool itself is the seeded part).
        selected[0] = 0;
        taken[0] = true;
        double[] minDistance = new double[pool.length];
        for (int i = 0; i < pool.length; i++) {
            minDistance[i] = distanceMetric.compute(pool[0], pool[i]);
        }

        // Both scans of a round are split into the same contiguous, ordered slices, so the
        // traversal stays bit-identical to the serial version while using every driver core.
        int[][] slices = sliceRanges(pool.length);

        int chosen = 1;
        while (chosen < targetCandidateCount) {
            // Argmax per slice in parallel, then combined in ASCENDING slice order so a tie
            // resolves to the lowest index — the serial tie rule, which the cluster ids depend on.
            double[][] sliceBest = sliceStream(slices)
                .mapToObj(slice -> {
                    int bestIndex = -1;
                    double bestDistance = Double.NEGATIVE_INFINITY;
                    for (int i = slices[slice][0]; i < slices[slice][1]; i++) {
                        if (!taken[i] && minDistance[i] > bestDistance) {
                            bestDistance = minDistance[i];
                            bestIndex = i;
                        }
                    }
                    return new double[] {bestDistance, bestIndex};
                })
                .toArray(double[][]::new);

            int farthest = -1;
            double best = Double.NEGATIVE_INFINITY;
            for (double[] candidate : sliceBest) {
                int index = (int) candidate[1];
                if (index >= 0 && candidate[0] > best) {
                    best = candidate[0];
                    farthest = index;
                }
            }
            // Only reachable when every remaining distance is NaN (NaN fails every '>'), so
            // report the cause instead of an index-out-of-bounds.
            if (farthest < 0) {
                throw new IllegalStateException(
                    "kcenter: all distances from the selected set are NaN after " + chosen
                    + " centres — the " + distanceMetric.getClass().getSimpleName()
                    + " data contains NaN/Inf coordinates (or a zero vector under cosine distance)");
            }
            selected[chosen] = farthest;
            taken[farthest] = true;

            // The update writes each index once and is order-independent, so it may run in
            // parallel without changing the result bit-for-bit.
            final int centre = farthest;
            sliceStream(slices).forEach(slice -> {
                for (int i = slices[slice][0]; i < slices[slice][1]; i++) {
                    double d = distanceMetric.compute(pool[centre], pool[i]);
                    if (d < minDistance[i]) {
                        minDistance[i] = d;
                    }
                }
            });
            chosen++;
        }

        double[][] out = new double[targetCandidateCount][];
        for (int i = 0; i < targetCandidateCount; i++) {
            out[i] = pool[selected[i]];
        }
        return out;
    }

    /** Contiguous, disjoint, ascending slices of {@code [0, length)} as {@code {from, until}};
     *  one slice below the parallel threshold. */
    static int[][] sliceRanges(int length) {
        int sliceCount = length < MinParallelPoolSize
            ? 1
            : Math.max(1, Runtime.getRuntime().availableProcessors());
        int sliceSize = (length + sliceCount - 1) / sliceCount;
        int used = 0;
        int[][] slices = new int[sliceCount][];
        for (int slice = 0; slice < sliceCount; slice++) {
            int from = slice * sliceSize;
            int until = Math.min(length, (slice + 1) * sliceSize);
            if (from < until) {
                slices[used++] = new int[] {from, until};
            }
        }
        return java.util.Arrays.copyOf(slices, used);
    }

    /** Slice indices, run in parallel once there is more than one slice. */
    private static IntStream sliceStream(int[][] slices) {
        IntStream stream = IntStream.range(0, slices.length);
        return slices.length > 1 ? stream.parallel() : stream;
    }
}
