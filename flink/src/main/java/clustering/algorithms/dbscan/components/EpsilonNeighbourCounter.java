package clustering.algorithms.dbscan.components;

import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.ManagedMemory;
import clustering.core.PointSource;
import clustering.core.WeightedPoint;
import clustering.core.WeightedPointTypeInfo;
import clustering.distance.DistanceMetric;
import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * Computes exact epsilon-neighborhood densities for candidate points.
 * Executes as a single bounded iteration where data points are cached once per subtask,
 * and candidate chunks cycle through the feedback edge.
 */
public final class EpsilonNeighbourCounter {

    private static final Logger logger = LoggerFactory.getLogger(EpsilonNeighbourCounter.class);

    private static final TypeInformation<IterationState> STATE_TYPE = TypeInformation.of(IterationState.class);
    private static final TypeInformation<PartialCounts> PARTIAL_TYPE = TypeInformation.of(PartialCounts.class);
    private static final TypeInformation<IterationUpdate> UPDATE_TYPE = TypeInformation.of(IterationUpdate.class);
    private static final TypeInformation<ChunkResult> CHUNK_RESULT_TYPE = TypeInformation.of(ChunkResult.class);

    private static final int MAX_CHUNK_SIZE = 100_000;

    private EpsilonNeighbourCounter() {}

    /**
     * Computes the number of dataset points within epsilon distance for each candidate.
     * Incorporates point weights to calculate density mass rather than raw row counts.
     */
    public static double[] computeNeighbourhoodDensities(
            PointSource source,
            EnvFactory envFactory,
            double[][] candidatePoints,
            double epsilon,
            DistanceMetric distanceMetric,
            int maxChunkSize) {

        if (candidatePoints.length == 0) {
            return new double[0];
        }

        int activeChunkSize = Math.max(1, Math.min(maxChunkSize, candidatePoints.length));
        if (activeChunkSize > MAX_CHUNK_SIZE) {
            logger.warn("epsilon-counting: chunkSize {} capped to {} to prevent Flink collect sink record size overflow.",
                activeChunkSize, MAX_CHUNK_SIZE);
            activeChunkSize = MAX_CHUNK_SIZE;
        }

        StreamExecutionEnvironment env = envFactory.newEnv();
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);

        DataStream<IterationState> initStateStream =
            env.fromCollection(Collections.singletonList(IterationState.createInitial(candidatePoints, activeChunkSize)), STATE_TYPE);
        DataStream<WeightedPoint> pointsStream = source.create(env);

        DataStreamList iterationResult = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initStateStream),
            ReplayableDataStreamList.notReplay(pointsStream),
            IterationConfig.newBuilder().build(),
            new CountingIterationBody(candidatePoints, activeChunkSize, epsilon, distanceMetric));

        double[] finalCounts = new double[candidatePoints.length];
        boolean[] chunkProcessedFlags = new boolean[candidatePoints.length];

        FlinkJobs.consume(iterationResult.<ChunkResult>get(0), "epsilon-counting", result -> {
            System.arraycopy(result.counts, 0, finalCounts, result.startIndex, result.counts.length);
            for (int i = 0; i < result.counts.length; i++) {
                chunkProcessedFlags[result.startIndex + i] = true;
            }
        });

        for (int i = 0; i < chunkProcessedFlags.length; i++) {
            if (!chunkProcessedFlags[i]) {
                throw new RuntimeException(
                    String.format("epsilon-counting failed: missing density count for candidate %d of %d.",
                        i, candidatePoints.length));
            }
        }
        return finalCounts;
    }

    private static final class CountingIterationBody implements IterationBody {
        private final double[][] candidatePoints;
        private final int chunkSize;
        private final double epsilon;
        private final DistanceMetric distanceMetric;

        CountingIterationBody(double[][] candidatePoints, int chunkSize, double epsilon, DistanceMetric distanceMetric) {
            this.candidatePoints = candidatePoints;
            this.chunkSize = chunkSize;
            this.epsilon = epsilon;
            this.distanceMetric = distanceMetric;
        }

        @Override
        public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<IterationState> stateStream = variableStreams.get(0);
            DataStream<WeightedPoint> pointsStream = dataStreams.get(0);

            DataStream<PartialCounts> partialCounts = pointsStream
                .connect(stateStream.broadcast())
                .transform("epsilon-count-chunk", PARTIAL_TYPE, new DistanceEvaluator(epsilon, distanceMetric));

            ManagedMemory.forPointCache(partialCounts);

            DataStream<IterationUpdate> updates = partialCounts
                .flatMap(new ResultsCombiner(candidatePoints, chunkSize))
                .setParallelism(1)
                .returns(UPDATE_TYPE);

            DataStream<IterationState> newStateStream = updates
                .map((MapFunction<IterationUpdate, IterationState>) update -> update.state)
                .returns(STATE_TYPE)
                .setParallelism(1);

            DataStream<Integer> terminationSignal = updates
                .flatMap(new TerminationEvaluator())
                .returns(Types.INT)
                .setParallelism(1);

            DataStream<ChunkResult> chunkResults = updates
                .map((MapFunction<IterationUpdate, ChunkResult>) update -> update.chunkResult)
                .returns(CHUNK_RESULT_TYPE)
                .setParallelism(1);

            return new IterationBodyResult(
                DataStreamList.of(newStateStream),
                DataStreamList.of(chunkResults),
                terminationSignal);
        }
    }

    /**
     * Subtask operator evaluating distances between cached points and the current candidate chunk.
     */
    private static final class DistanceEvaluator
            extends AbstractStreamOperator<PartialCounts>
            implements TwoInputStreamOperator<WeightedPoint, IterationState, PartialCounts>,
                       IterationListener<PartialCounts> {

        private final double epsilon;
        private final DistanceMetric distanceMetric;
        private transient ListStateWithCache<WeightedPoint> cachedPoints;
        private transient ListState<IterationState> iterationStateList;

        DistanceEvaluator(double epsilon, DistanceMetric distanceMetric) {
            this.epsilon = epsilon;
            this.distanceMetric = distanceMetric;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            iterationStateList = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("epsilon-count-state", STATE_TYPE));
            cachedPoints = new ListStateWithCache<>(
                WeightedPointTypeInfo.INSTANCE.createSerializer(getExecutionConfig()),
                getContainingTask(),
                getRuntimeContext(),
                context,
                config.getOperatorID());
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            cachedPoints.snapshotState(context);
        }

        @Override
        public void processElement1(StreamRecord<WeightedPoint> record) throws Exception {
            cachedPoints.add(record.getValue());
        }

        @Override
        public void processElement2(StreamRecord<IterationState> record) throws Exception {
            iterationStateList.add(record.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<PartialCounts> out) throws Exception {
            Optional<IterationState> currentStateOpt = OperatorStateUtils.getUniqueElement(iterationStateList, "epsilon-count-state");
            if (!currentStateOpt.isPresent()) {
                return;
            }

            IterationState state = currentStateOpt.get();
            double[][] candidateChunk = state.candidateChunk;
            double[] localCounts = new double[candidateChunk.length];

            for (WeightedPoint point : cachedPoints.get()) {
                double[] coordinates = point.features.values;
                double weight = point.weight;

                for (int i = 0; i < candidateChunk.length; i++) {
                    if (distanceMetric.withinRadius(candidateChunk[i], coordinates, epsilon)) {
                        localCounts[i] += weight;
                    }
                }
            }

            PartialCounts partial = new PartialCounts();
            partial.subtaskId = getRuntimeContext().getIndexOfThisSubtask();
            partial.counts = localCounts;
            partial.startIndex = state.startIndex;
            out.collect(partial);
            iterationStateList.clear();
        }

        @Override
        public void onIterationTerminated(Context context, Collector<PartialCounts> out) throws Exception {
            cachedPoints.clear();
        }

        @Override public void processWatermark1(Watermark mark) {}
        @Override public void processWatermark2(Watermark mark) {}
        @Override public void processLatencyMarker1(LatencyMarker latencyMarker) {}
        @Override public void processLatencyMarker2(LatencyMarker latencyMarker) {}
    }

    /**
     * Combiner merging local counts for the active chunk and preparing the next chunk iteration.
     */
    private static final class ResultsCombiner
            implements FlatMapFunction<PartialCounts, IterationUpdate>, IterationListener<IterationUpdate> {

        private final double[][] allCandidates;
        private final int maxChunkSize;
        private transient List<PartialCounts> resultsBuffer;
        private transient long startTimeNanos;
        private transient int processedChunks;

        ResultsCombiner(double[][] allCandidates, int maxChunkSize) {
            this.allCandidates = allCandidates;
            this.maxChunkSize = maxChunkSize;
        }

        @Override
        public void flatMap(PartialCounts partial, Collector<IterationUpdate> out) {
            if (resultsBuffer == null) {
                resultsBuffer = new ArrayList<>();
            }
            resultsBuffer.add(partial);
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<IterationUpdate> out) {
            if (resultsBuffer == null || resultsBuffer.isEmpty()) {
                return;
            }
            if (processedChunks == 0) {
                startTimeNanos = System.nanoTime();
            }

            resultsBuffer.sort(Comparator.comparingInt(p -> p.subtaskId));
            int chunkStartIndex = resultsBuffer.get(0).startIndex;
            int chunkLength = resultsBuffer.get(0).counts.length;
            double[] aggregatedCounts = new double[chunkLength];

            for (PartialCounts partial : resultsBuffer) {
                for (int i = 0; i < chunkLength; i++) {
                    aggregatedCounts[i] += partial.counts[i];
                }
            }
            resultsBuffer = null;

            int nextChunkStart = chunkStartIndex + chunkLength;
            processedChunks++;
            int totalChunks = (allCandidates.length + maxChunkSize - 1) / maxChunkSize;
            double elapsedSeconds = (System.nanoTime() - startTimeNanos) / 1e9;

            logger.info(String.format("epsilon-counting: chunk %d/%d (%d candidates), %.0f s elapsed, %.0f s left",
                processedChunks, totalChunks, chunkLength, elapsedSeconds,
                elapsedSeconds * (totalChunks - processedChunks) / processedChunks));

            IterationUpdate update = new IterationUpdate();
            update.chunkResult = ChunkResult.create(chunkStartIndex, aggregatedCounts);

            if (nextChunkStart >= allCandidates.length) {
                update.state = IterationState.create(allCandidates, chunkStartIndex, chunkLength);
                update.shouldStop = true;
            } else {
                update.state = IterationState.create(allCandidates, nextChunkStart, maxChunkSize);
                update.shouldStop = false;
            }
            out.collect(update);
        }

        @Override
        public void onIterationTerminated(Context context, Collector<IterationUpdate> out) {}
    }

    private static final class TerminationEvaluator
            implements FlatMapFunction<IterationUpdate, Integer>, IterationListener<Integer> {
        private transient boolean shouldStop;

        @Override
        public void flatMap(IterationUpdate update, Collector<Integer> out) {
            shouldStop = update.shouldStop;
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<Integer> out) {
            if (!shouldStop) {
                out.collect(0);
            }
            shouldStop = false;
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Integer> out) {}
    }

    public static final class IterationState implements Serializable {
        public int startIndex;
        public double[][] candidateChunk;

        public IterationState() {}

        static IterationState createInitial(double[][] candidates, int chunkSize) {
            return create(candidates, 0, chunkSize);
        }

        static IterationState create(double[][] candidates, int startIdx, int chunkSize) {
            IterationState state = new IterationState();
            state.startIndex = startIdx;
            state.candidateChunk = Arrays.copyOfRange(candidates, startIdx,
                Math.min(candidates.length, startIdx + chunkSize));
            return state;
        }
    }

    public static final class PartialCounts implements Serializable {
        public int subtaskId;
        public int startIndex;
        public double[] counts;

        public PartialCounts() {}
    }

    public static final class IterationUpdate implements Serializable {
        public ChunkResult chunkResult;
        public IterationState state;
        public boolean shouldStop;

        public IterationUpdate() {}
    }

    public static final class ChunkResult implements Serializable {
        public int startIndex;
        public double[] counts;

        public ChunkResult() {}

        static ChunkResult create(int startIndex, double[] counts) {
            ChunkResult result = new ChunkResult();
            result.startIndex = startIndex;
            result.counts = counts;
            return result;
        }
    }
}
