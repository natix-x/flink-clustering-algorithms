package clustering.algorithms.dbscan.components;

import clustering.algorithms.dbscan.utils.ExecutionPlan;
import clustering.algorithms.dbscan.utils.ScanProgress;
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

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Computes connected components of the ε-graph over core points.
 * Employs either a driver-local parallel scan or a distributed Flink job
 * depending on available resources and graph size.
 */
public final class EpsilonGraphComponents {

    private static final Logger logger = LoggerFactory.getLogger(EpsilonGraphComponents.class);

    private static final int DRIVER_ROWS_PER_BLOCK = 128;
    private static final double COMPARISONS_PER_CORE_SECOND = 2.5e8;
    private static final double MIN_DRIVER_SECONDS_FOR_CLUSTER = 30.0;
    private static final int MIN_PARALLELISM_RATIO = 2;
    private static final long MAX_SHIPPED_COORDS_BYTES = 64L * 1024 * 1024;

    private EpsilonGraphComponents() {}

    /**
     * Constructs the ε-graph and identifies connected components.
     * Components are assigned contiguous IDs starting from 0.
     */
    public static int[] compute(
            double[][] corePoints,
            double epsilon,
            DistanceMetric distanceMetric,
            EnvFactory envFactory,
            int parallelism,
            double estimatedCoreDegree) {

        int dimensionality = getDimension(corePoints);
        long payloadBytes = (long) corePoints.length * dimensionality * 8L;
        ExecutionPlan executionPlan = determinePlan(corePoints.length, dimensionality, parallelism,
            Runtime.getRuntime().availableProcessors(), payloadBytes);

        long startTimeNanos = System.nanoTime();
        int[] componentLabels;

        if (executionPlan instanceof ExecutionPlan.Distributed) {
            double estimatedEdges = corePoints.length * Math.max(0.0, estimatedCoreDegree) / 2.0;
            logger.info(String.format(
                "dbscanpp: step 3 ε-graph distributed — %d cores, expected degree %.0f "
                + "(~%.0f edges, ~%.0f MB streamed), payload %.0f MB",
                corePoints.length, estimatedCoreDegree, estimatedEdges, estimatedEdges * 4 / 1e6, payloadBytes / 1e6));
            componentLabels = executeDistributedScan(corePoints, epsilon, distanceMetric, envFactory, parallelism);
        } else {
            logger.info("dbscanpp: step 3 ε-graph local — {}", ((ExecutionPlan.DriverLocal) executionPlan).reason);
            componentLabels = executeLocalScan(corePoints, epsilon, distanceMetric);
        }

        logger.info(String.format("dbscanpp: step 3 ε-graph completed in %.0f s over %d cores",
            (System.nanoTime() - startTimeNanos) / 1e9, corePoints.length));
        return componentLabels;
    }

    /** Decides whether to run the scan locally on the driver or submit a Flink job. */
    public static ExecutionPlan determinePlan(int numCores, int dimension, int clusterParallelism, int driverCores,
                        long payloadBytes) {

        double totalPairs = (double) numCores * (numCores - 1) / 2.0;
        double estimatedDriverSeconds =
            totalPairs * Math.max(1, dimension) / (Math.max(1, driverCores) * COMPARISONS_PER_CORE_SECOND);

        if (estimatedDriverSeconds < MIN_DRIVER_SECONDS_FOR_CLUSTER) {
            return new ExecutionPlan.DriverLocal(String.format(
                "Estimated local time ~%.0f s is below threshold %.0f s",
                estimatedDriverSeconds, MIN_DRIVER_SECONDS_FOR_CLUSTER));
        }
        if (clusterParallelism < MIN_PARALLELISM_RATIO * Math.max(1, driverCores)) {
            return new ExecutionPlan.DriverLocal(String.format(
                "Cluster parallelism %d does not justify overhead against %d driver cores",
                clusterParallelism, driverCores));
        }
        if (payloadBytes > MAX_SHIPPED_COORDS_BYTES) {
            return new ExecutionPlan.DriverLocal(String.format(
                "Payload size %.1f GB exceeds maximum %.1f GB",
                payloadBytes / 1e9, MAX_SHIPPED_COORDS_BYTES / 1e9));
        }
        return new ExecutionPlan.Distributed();
    }

    /** Executes a multi-core ε-graph scan on the driver. */
    public static int[] executeLocalScan(double[][] corePoints, double epsilon, DistanceMetric distanceMetric) {
        UnionFind componentTracker = new UnionFind(corePoints.length);
        ScanProgress progressTracker = new ScanProgress(corePoints.length);
        int blockStartIndex = 0;

        while (blockStartIndex < corePoints.length) {
            int blockEndIndex = Math.min(blockStartIndex + DRIVER_ROWS_PER_BLOCK, corePoints.length);
            final int startIdx = blockStartIndex;

            int[][] blockNeighbors = IntStream.range(startIdx, blockEndIndex).parallel()
                .mapToObj(rowIdx -> findNeighborsAbove(rowIdx, corePoints, epsilon, distanceMetric))
                .toArray(int[][]::new);

            for (int rowIdx = blockStartIndex; rowIdx < blockEndIndex; rowIdx++) {
                mergeEdges(componentTracker, rowIdx, blockNeighbors[rowIdx - blockStartIndex]);
            }
            blockStartIndex = blockEndIndex;
            progressTracker.report(blockStartIndex, "driver-local");
        }
        return componentTracker.componentIds();
    }

    /**
     * Submits a Flink job to stream ε-graph edges back to the driver.
     * Balances work by pairing short and long rows.
     */
    public static int[] executeDistributedScan(double[][] corePoints, double epsilon, DistanceMetric distanceMetric,
                             EnvFactory envFactory, int parallelism) {

        StreamExecutionEnvironment env = envFactory.newEnv();
        env.setParallelism(Math.max(1, parallelism));

        DataStream<int[]> edgeStream = env
            .fromSequence(0L, corePoints.length - 1L)
            .map(new RowScanMapper(corePoints, epsilon, distanceMetric))
            .returns(PrimitiveArrayTypeInfo.INT_PRIMITIVE_ARRAY_TYPE_INFO)
            .name("epsilon-graph-scan");

        UnionFind componentTracker = new UnionFind(corePoints.length);
        ScanProgress progressTracker = new ScanProgress(corePoints.length);
        long[] completedRows = {0L};

        FlinkJobs.consume(edgeStream, "dbscanpp-epsilon-graph", edgeRecord -> {
            int sourceRow = edgeRecord[0];
            for (int t = 1; t < edgeRecord.length; t++) {
                componentTracker.union(sourceRow, edgeRecord[t]);
            }
            completedRows[0]++;
            progressTracker.report((int) completedRows[0], "distributed");
        });

        return componentTracker.componentIds();
    }

    private static final class RowScanMapper implements MapFunction<Long, int[]> {
        private final double[][] corePoints;
        private final double epsilon;
        private final DistanceMetric distanceMetric;

        RowScanMapper(double[][] corePoints, double epsilon, DistanceMetric distanceMetric) {
            this.corePoints = corePoints;
            this.epsilon = epsilon;
            this.distanceMetric = distanceMetric;
        }

        @Override
        public int[] map(Long sequenceIndex) {
            int targetRow = mapBalancedRow(sequenceIndex, corePoints.length);
            int[] neighbors = findNeighborsAbove(targetRow, corePoints, epsilon, distanceMetric);
            int[] edgeRecord = new int[neighbors.length + 1];
            edgeRecord[0] = targetRow;
            System.arraycopy(neighbors, 0, edgeRecord, 1, neighbors.length);
            return edgeRecord;
        }
    }

    /** Maps a sequence index to a row index such that workload is evenly distributed across tasks. */
    public static int mapBalancedRow(long sequenceIndex, int totalRows) {
        long halfIndex = sequenceIndex / 2;
        return (sequenceIndex % 2 == 0) ? (int) halfIndex : (int) (totalRows - 1 - halfIndex);
    }

    private static int[] findNeighborsAbove(int sourceRow, double[][] corePoints, double epsilon, DistanceMetric distanceMetric) {
        int[] neighborBuffer = new int[corePoints.length - sourceRow - 1];
        int count = 0;
        for (int targetRow = sourceRow + 1; targetRow < corePoints.length; targetRow++) {
            if (distanceMetric.withinRadius(corePoints[sourceRow], corePoints[targetRow], epsilon)) {
                neighborBuffer[count++] = targetRow;
            }
        }
        return Arrays.copyOf(neighborBuffer, count);
    }

    private static void mergeEdges(UnionFind tracker, int sourceRow, int[] neighbors) {
        for (int neighbor : neighbors) {
            tracker.union(sourceRow, neighbor);
        }
    }

    private static int getDimension(double[][] points) {
        return points.length == 0 ? 0 : points[0].length;
    }
}
