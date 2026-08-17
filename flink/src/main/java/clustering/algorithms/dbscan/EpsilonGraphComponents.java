package clustering.algorithms.dbscan;

import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.distance.DistanceMetric;
import clustering.utils.UnionFind;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.stream.IntStream;

/** Connected components of the ε-graph over the core points — step 3 of DBSCAN++.
 *
 *  Two halves with very different natures:
 *  <ul>
 *    <li><b>finding the edges</b>, m²/2 distance computations, embarrassingly parallel;</li>
 *    <li><b>the union-find</b>, O(m·α(m)), which stays on the driver ({@link UnionFind} is not
 *        thread-safe, and a distributed connected-components pass is the one thing the density
 *        slot is designed to avoid).</li>
 *  </ul>
 *  The union-find is free next to the m² scan, so parallelising the first half is the whole win.
 *  Two implementations, same labels:
 *  <ul>
 *    <li>{@link #local} — parallel over the DRIVER's cores. Produces the edges of a small row
 *        block and consumes them immediately: peak memory is one block, nothing is serialised.</li>
 *    <li>{@link #distributed} — ONE Flink job that scans the row space across the cluster and
 *        STREAMS its edges back; the driver unions them as they arrive. Wins when the cluster has
 *        materially more cores than the driver AND the scan is long enough to amortise a job —
 *        both measured, see {@link #planFor}.</li>
 *  </ul>
 *
 *  <h3>Why the Flink version has no block loop — and no dense-graph fallback</h3>
 *  The Spark side must cut this scan into row BLOCKS and size each block from an estimate of the
 *  edge count, because {@code collect} materialises a whole block's edges on the driver: at
 *  4096 rows × ~10⁴ neighbours that is ~10⁸ ints per task, which OOM-ed in {@code Kryo.toBytes}
 *  before the first block finished. So on Spark a dense ε-graph is refused the cluster and falls
 *  back to the driver.
 *
 *  Flink's collect is a back-pressured ITERATOR ({@link FlinkJobs#consume}), so the edges are
 *  folded into the union-find as they stream in and the driver never holds more than one row's
 *  worth. There is nothing to block, nothing to size, and no density at which the phase must give
 *  up the cluster — the estimated degree survives here only as a log line saying how much edge
 *  traffic to expect. That difference is a property of the two engines' result-collection models,
 *  not of the algorithm, and it is worth stating as a finding.
 *
 *  <h3>Why the labels are identical either way</h3>
 *  {@link UnionFind#union} attaches the larger root under the smaller, so a component's
 *  representative is its minimum member index no matter what order the edges arrive in, and
 *  {@code componentIds} numbers components by that minimum. Connected components are a property
 *  of the edge SET, so any order of the same edges gives the same labels — which is why the
 *  distributed path may stream edges in whatever order the subtasks finish, and why
 *  reproducibility does not depend on parallelism here. This is the one phase of the whole repo
 *  that is order-insensitive by construction rather than by pinning an order. */
final class EpsilonGraphComponents {

    private static final Logger logger = LoggerFactory.getLogger(EpsilonGraphComponents.class);

    /** Rows scanned per block in the driver-local path. Bounds the materialised edge list at
     *  {@code RowsPerBlock · m} ints instead of the O(m²) a single pass would need. */
    private static final int RowsPerBlock = 128;

    /** Measured throughput of the scan, in coordinate-comparisons per core-second — the yardstick
     *  for "would the driver be busy long enough to be worth a job". Calibrated on this repo's own
     *  scan (m = 100 000, 3-D, 12 driver cores: 5.0 s for 1.5·10^10 coordinate comparisons) and
     *  rounded down, so the estimate errs towards staying local. */
    private static final double ComparisonsPerCoreSecond = 2.5e8;

    /** The driver has to be busy for at least this long before a job is worth launching. The fixed
     *  cost is not small: measured on ONE 12-core machine, the distributed path took 21.9 s where
     *  the driver-local one took 5.0 s (m = 100 000, sparse graph) — cluster/job startup, one
     *  record per row through the network stack, and the collect iterator. That measurement is
     *  also why the ratio guard below exists: extra workers, not extra rows, are what make this
     *  phase worth distributing. */
    private static final double MinDriverSecondsToDistribute = 30.0;

    /** Required ratio of cluster parallelism to DRIVER cores. The driver-local path already uses
     *  every core of the machine the driver runs on, so a job only adds compute when the cluster
     *  has substantially more of it — on a single-node run (parallelism ≈ driver cores) it can only
     *  add overhead. */
    private static final int MinParallelismRatio = 2;

    /** Cap on the core coordinates shipped with the job. Every subtask compares its rows against
     *  ALL cores, so the array travels inside the JobGraph (Flink's DataStream API has no
     *  broadcast-variable channel) — a far tighter budget than a Spark broadcast, and the reason
     *  the distributed path is refused rather than left to fail in job submission. At the sizes
     *  above this cap the m²·d scan is hopeless anyway, so the honest answer is a smaller `s`. */
    private static final long MaxShippedCoordsBytes = 64L * 1024 * 1024;

    /** Minimum gap between two progress lines. The scan produces no output of its own, so without
     *  these a long run is indistinguishable from a hang. */
    private static final long LogIntervalNanos = 30L * 1000_000_000L;

    private EpsilonGraphComponents() {}

    /** Which path to run, and — when it is the driver — why, since "step 3 took X s" means
     *  different things on one machine and on a cluster. */
    abstract static class Plan {
        static final class Distributed extends Plan {
            @Override public String toString() {
                return "Distributed";
            }
        }

        static final class DriverLocal extends Plan {
            final String reason;

            DriverLocal(String reason) {
                this.reason = reason;
            }

            @Override public String toString() {
                return "DriverLocal(" + reason + ")";
            }
        }
    }

    /** Cluster id per core point, contiguous from 0, numbered by ascending minimum member index.
     *
     *  @param expectedDegree expected number of ε-neighbours of a core point WITHIN the core set.
     *         Step 2 already measured exact degrees against the full data, so this costs nothing
     *         to supply. Unlike on Spark it does not gate the path (see the class docs) — it only
     *         reports the edge traffic the distributed path is about to generate. */
    static int[] compute(
            double[][] cores,
            double eps,
            DistanceMetric metric,
            EnvFactory envs,
            int parallelism,
            double expectedDegree) {
        int dimension = dimensionOf(cores);
        long shippedBytes = (long) cores.length * dimension * 8L;
        Plan plan = planFor(cores.length, dimension, parallelism,
            Runtime.getRuntime().availableProcessors(), shippedBytes);

        long startedAt = System.nanoTime();
        int[] labels;
        if (plan instanceof Plan.Distributed) {
            // Half the expected degree: a row emits only its j > i neighbours.
            double expectedEdges = cores.length * Math.max(0.0, expectedDegree) / 2.0;
            logger.info(String.format(
                "dbscanpp: step 3 ε-graph distributed — %d cores, expected degree %.0f "
                + "(~%.0f edges, ~%.0f MB streamed back), coords %.0f MB shipped with the job",
                cores.length, expectedDegree, expectedEdges, expectedEdges * 4 / 1e6, shippedBytes / 1e6));
            labels = distributed(cores, eps, metric, envs, parallelism);
        } else {
            logger.info("dbscanpp: step 3 ε-graph on the driver — {}", ((Plan.DriverLocal) plan).reason);
            labels = local(cores, eps, metric);
        }
        logger.info(String.format("dbscanpp: step 3 ε-graph done in %.0f s over %d cores",
            (System.nanoTime() - startedAt) / 1e9, cores.length));
        return labels;
    }

    /** Chooses the path. Pure, so the decision is unit-testable without a cluster.
     *
     *  Distributes only when the cluster can actually help, which takes THREE things and not just
     *  a big m:
     *  <ol>
     *    <li>the driver would be busy for at least {@link #MinDriverSecondsToDistribute} — below
     *        that the job's fixed cost dominates;</li>
     *    <li>the cluster has {@link #MinParallelismRatio}× more parallelism than the driver has
     *        cores — the local path already uses all of them, so on a single-node run a job adds
     *        overhead and nothing else. Measured: 21.9 s distributed vs 5.0 s local on one 12-core
     *        machine;</li>
     *    <li>the core coordinates fit in the JobGraph.</li>
     *  </ol>
     *  Notably there is NO density condition — see the class docs for why Spark needs one and this
     *  does not.
     *
     *  @param parallelism cluster parallelism the run was configured with
     *  @param driverCores cores available to the driver-local path */
    static Plan planFor(int coreCount, int dimension, int parallelism, int driverCores,
                        long shippedCoordsBytes) {
        double pairs = (double) coreCount * (coreCount - 1) / 2.0;
        double driverSeconds =
            pairs * Math.max(1, dimension) / (Math.max(1, driverCores) * ComparisonsPerCoreSecond);
        if (driverSeconds < MinDriverSecondsToDistribute) {
            return new Plan.DriverLocal(String.format(
                "%d cores scan in ~%.0f s on %d driver cores (below the %.0f s a job is worth)",
                coreCount, driverSeconds, driverCores, MinDriverSecondsToDistribute));
        }
        if (parallelism < MinParallelismRatio * Math.max(1, driverCores)) {
            return new Plan.DriverLocal(String.format(
                "cluster parallelism %d is not %d× the driver's %d cores, so distributing the scan "
                + "would only add job and network overhead",
                parallelism, MinParallelismRatio, driverCores));
        }
        if (shippedCoordsBytes > MaxShippedCoordsBytes) {
            return new Plan.DriverLocal(String.format(
                "shipping %d cores with the job would take %.1f GB (cap %.1f GB) — lower "
                + "coreSampleFraction if this phase dominates",
                coreCount, shippedCoordsBytes / 1e9, MaxShippedCoordsBytes / 1e9));
        }
        return new Plan.Distributed();
    }

    /** Parallel over the driver's cores. Unions run after each block, sequentially. */
    static int[] local(double[][] cores, double eps, DistanceMetric metric) {
        UnionFind unionFind = new UnionFind(cores.length);
        ScanProgress progress = new ScanProgress(cores.length);
        int blockStart = 0;

        while (blockStart < cores.length) {
            int blockEnd = Math.min(blockStart + RowsPerBlock, cores.length);
            final int from = blockStart;
            int[][] neighboursPerRow = IntStream.range(from, blockEnd).parallel()
                .mapToObj(row -> neighboursAbove(row, cores, eps, metric))
                .toArray(int[][]::new);

            for (int row = blockStart; row < blockEnd; row++) {
                applyEdges(unionFind, row, neighboursPerRow[row - blockStart]);
            }
            blockStart = blockEnd;
            progress.report(blockStart, "driver-local");
        }
        return unionFind.componentIds();
    }

    /** ONE Flink job: every subtask scans a slice of the row space against the (shipped) core set
     *  and emits one record per row — {@code [row, j1, j2, ...]} — which the driver folds into the
     *  union-find as it streams in ({@link FlinkJobs#consume}). No blocks, no edge list, no
     *  size estimate: the driver's peak is one row's neighbours.
     *
     *  Row i scans m − i others, so the raw index range is wildly uneven work. The indices are
     *  therefore paired before scanning — index t becomes row {@code t/2} or {@code m-1-t/2} —
     *  so any contiguous slice of the sequence holds roughly the same amount of work, and Flink's
     *  even split of {@code fromSequence} becomes an even split of the SCAN. Cheap, stateless and
     *  deterministic; it changes which subtask does what, never the edge set. */
    static int[] distributed(double[][] cores, double eps, DistanceMetric metric,
                             EnvFactory envs, int parallelism) {
        StreamExecutionEnvironment env = envs.newEnv();
        env.setParallelism(Math.max(1, parallelism));
        DataStream<int[]> rowEdges = env
            .fromSequence(0L, cores.length - 1L)
            .map(new RowScan(cores, eps, metric))
            .returns(PrimitiveArrayTypeInfo.INT_PRIMITIVE_ARRAY_TYPE_INFO)
            .name("epsilon-graph-scan");

        UnionFind unionFind = new UnionFind(cores.length);
        ScanProgress progress = new ScanProgress(cores.length);
        long[] rowsDone = {0L};
        FlinkJobs.consume(rowEdges, "dbscanpp-epsilon-graph", record -> {
            int row = record[0];
            for (int t = 1; t < record.length; t++) {
                unionFind.union(row, record[t]);
            }
            rowsDone[0]++;
            progress.report((int) rowsDone[0], "distributed");
        });
        return unionFind.componentIds();
    }

    /** Scans one row of the ε-graph. The core coordinates are an operator field, shipped once with
     *  the job; the emitted record is {@code [row, neighbours above row...]}. */
    private static final class RowScan implements MapFunction<Long, int[]> {
        private final double[][] cores;
        private final double eps;
        private final DistanceMetric metric;

        RowScan(double[][] cores, double eps, DistanceMetric metric) {
            this.cores = cores;
            this.eps = eps;
            this.metric = metric;
        }

        @Override
        public int[] map(Long index) {
            int row = balancedRow(index, cores.length);
            int[] neighbours = neighboursAbove(row, cores, eps, metric);
            int[] record = new int[neighbours.length + 1];
            record[0] = row;
            System.arraycopy(neighbours, 0, record, 1, neighbours.length);
            return record;
        }
    }

    /** Pairs a short row with a long one so equal-sized index slices carry equal work. A bijection
     *  on {@code [0, m)}: even indices walk up from 0, odd indices walk down from m − 1. */
    static int balancedRow(long index, int coreCount) {
        long half = index / 2;
        return (index % 2 == 0) ? (int) half : (int) (coreCount - 1 - half);
    }

    /** Indices {@code j > row} within ε of {@code row}. Ascending by construction. */
    private static int[] neighboursAbove(int row, double[][] cores, double eps, DistanceMetric metric) {
        int[] buffer = new int[cores.length - row - 1];
        int found = 0;
        for (int j = row + 1; j < cores.length; j++) {
            if (metric.withinRadius(cores[row], cores[j], eps)) {
                buffer[found++] = j;
            }
        }
        return java.util.Arrays.copyOf(buffer, found);
    }

    private static void applyEdges(UnionFind unionFind, int row, int[] neighbours) {
        for (int neighbour : neighbours) {
            unionFind.union(row, neighbour);
        }
    }

    private static int dimensionOf(double[][] cores) {
        return cores.length == 0 ? 0 : cores[0].length;
    }

    /** Time-throttled progress over the m²/2 pair space.
     *
     *  Throttled by time rather than by block count: a block is milliseconds at m = 1 000 and
     *  minutes at m = 2·10⁶, so any fixed stride is either silent or spam. Progress counts PAIRS,
     *  since row i scans m − i others and a row percentage would run far ahead of the truth. In
     *  the distributed path rows arrive paired short-with-long, so the pair count is a fair
     *  estimate there too. */
    private static final class ScanProgress {
        private final int totalRows;
        private final double totalPairs;
        private final long startedAt = System.nanoTime();
        private long lastLogAt = startedAt;

        ScanProgress(int totalRows) {
            this.totalRows = totalRows;
            this.totalPairs = (double) totalRows * totalRows / 2;
        }

        void report(int rowsDone, String path) {
            long now = System.nanoTime();
            if (now - lastLogAt >= LogIntervalNanos) {
                lastLogAt = now;
                double elapsed = (now - startedAt) / 1e9;
                double done = ((double) rowsDone * totalRows - (double) rowsDone * rowsDone / 2) / totalPairs;
                logger.info(String.format(
                    "dbscanpp: step 3 ε-graph (%s) %.1f%% (%d/%d rows), %.0f s elapsed, %.0f s left",
                    path, done * 100, rowsDone, totalRows, elapsed, elapsed * (1 - done) / done));
            }
        }
    }
}
