package clustering.algorithms.dbscan.components;

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

/**
 * Computes connected components of the ε-graph over core points via a distributed Flink job.
 */
public final class EpsilonGraphComponents {

    private static final Logger logger = LoggerFactory.getLogger(EpsilonGraphComponents.class);

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
        double estimatedEdges = corePoints.length * Math.max(0.0, estimatedCoreDegree) / 2.0;
        logger.info(String.format(
            "dbscanpp: step 3 ε-graph distributed — %d cores, expected degree %.0f "
            + "(~%.0f edges, ~%.0f MB streamed), payload %.0f MB",
            corePoints.length, estimatedCoreDegree, estimatedEdges, estimatedEdges * 4 / 1e6, payloadBytes / 1e6));

        long startTimeNanos = System.nanoTime();
        int[] componentLabels = distributed(corePoints, epsilon, distanceMetric, envFactory, parallelism);

        logger.info(String.format("dbscanpp: step 3 ε-graph completed in %.0f s over %d cores",
            (System.nanoTime() - startTimeNanos) / 1e9, corePoints.length));
        return componentLabels;
    }

    /**
     * Submits a Flink job to stream ε-graph edges back to the driver.
     * Balances work by pairing short and long rows.
     */
    public static int[] distributed(double[][] corePoints, double epsilon, DistanceMetric distanceMetric,
                             EnvFactory envFactory, int parallelism) {

        StreamExecutionEnvironment env = envFactory.newEnv();
        env.setParallelism(Math.max(1, parallelism));

        DataStream<int[]> edgeStream = env
            .fromSequence(0L, corePoints.length - 1L)
            .map(new RowScanMapper(corePoints, epsilon, distanceMetric))
            .returns(PrimitiveArrayTypeInfo.INT_PRIMITIVE_ARRAY_TYPE_INFO)
            .name("epsilon-graph-scan");

        UnionFind componentTracker = new UnionFind(corePoints.length);

        FlinkJobs.consume(edgeStream, "dbscanpp-epsilon-graph", edgeRecord -> {
            int sourceRow = edgeRecord[0];
            for (int t = 1; t < edgeRecord.length; t++) {
                componentTracker.union(sourceRow, edgeRecord[t]);
            }
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
            int targetRow = balancedRow(sequenceIndex, corePoints.length);
            int[] neighbors = findNeighborsAbove(targetRow, corePoints, epsilon, distanceMetric);
            int[] edgeRecord = new int[neighbors.length + 1];
            edgeRecord[0] = targetRow;
            System.arraycopy(neighbors, 0, edgeRecord, 1, neighbors.length);
            return edgeRecord;
        }
    }

    /** Maps a sequence index to a row index such that workload is evenly distributed across tasks. */
    public static int balancedRow(long sequenceIndex, int totalRows) {
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

    private static int getDimension(double[][] points) {
        return points.length == 0 ? 0 : points[0].length;
    }
}
