package clustering.algorithms.kmedoids.distributed;

import clustering.algorithms.kmeans.CentroidIteration;
import clustering.algorithms.kmedoids.KMedoidsModel;
import clustering.algorithms.kmedoids.components.SwapMove;
import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Clusterer;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.ManagedMemory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.core.WeightedPoint;
import clustering.core.WeightedPointTypeInfo;
import clustering.core.Points;
import clustering.distance.DistanceMetric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * Distributed FastPAM training (BUILD + SWAP) executed as a single Flink ML bounded-iteration job.
 * Avoids per-round job submission overhead by caching points locally and cycling medoid state.
 */
public class DistributedFastPAM implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(DistributedFastPAM.class);

    private static final TypeInformation<IterationState> STATE_TYPE = TypeInformation.of(IterationState.class);
    private static final TypeInformation<PartialStats> PARTIAL_TYPE = TypeInformation.of(PartialStats.class);
    private static final TypeInformation<IterationUpdate> UPDATE_TYPE = TypeInformation.of(IterationUpdate.class);

    private final int targetK;
    private final int maxIterations;
    private final DistanceMetric distanceMetric;

    public DistributedFastPAM(int targetK, int maxIterations, DistanceMetric distanceMetric) {
        this.targetK = targetK;
        this.maxIterations = maxIterations;
        this.distanceMetric = distanceMetric;
    }

    @Override
    public Model fit(PointSource source, EnvFactory envFactory, int parallelism) {
        double[][] candidateCoords = Points.toArray(Datasets.collectAll(source, envFactory));
        int numCandidates = candidateCoords.length;

        if (numCandidates < targetK) {
            throw new IllegalArgumentException(
                "Dataset too small: n=" + numCandidates + " points but k=" + targetK + " medoids requested.");
        }

        StreamExecutionEnvironment env = envFactory.newEnv();
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);

        DataStream<IterationState> initStateStream =
            env.fromCollection(java.util.Collections.singletonList(IterationState.initialize(targetK)), STATE_TYPE);
        DataStream<WeightedPoint> pointsStream = source.create(env);

        DataStreamList iterationResult = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initStateStream),
            ReplayableDataStreamList.notReplay(pointsStream),
            IterationConfig.newBuilder().build(),
            new FastPamIterationBody(candidateCoords, targetK, maxIterations, distanceMetric));

        IterationState finalState = FlinkJobs.last(iterationResult.<IterationState>get(0), "distfastpam-fit");

        int[] finalMedoids = (finalState == null) ? generateInitialIndices(numCandidates, targetK) : finalState.medoids;
        DenseVector[] medoidVectors = new DenseVector[targetK];
        for (int i = 0; i < targetK; i++) {
            medoidVectors[i] = new DenseVector(candidateCoords[finalMedoids[i]]);
        }
        return new KMedoidsModel(medoidVectors, distanceMetric);
    }

    private static final class FastPamIterationBody implements IterationBody {
        private final double[][] candidateCoords;
        private final int targetK;
        private final int maxIterations;
        private final DistanceMetric distanceMetric;

        FastPamIterationBody(double[][] candidateCoords, int targetK, int maxIterations, DistanceMetric distanceMetric) {
            this.candidateCoords = candidateCoords;
            this.targetK = targetK;
            this.maxIterations = maxIterations;
            this.distanceMetric = distanceMetric;
        }

        @Override
        public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<IterationState> stateStream = variableStreams.get(0);
            DataStream<WeightedPoint> pointsStream = dataStreams.get(0);

            DataStream<PartialStats> partialStats = pointsStream
                .connect(stateStream.broadcast())
                .transform("dfastpam-partial", PARTIAL_TYPE, new PartialAggregator(candidateCoords, targetK, distanceMetric));

            ManagedMemory.forPointCache(partialStats);

            DataStream<IterationUpdate> updates = partialStats
                .flatMap(new StatsCombiner(candidateCoords.length, targetK, maxIterations))
                .setParallelism(1)
                .returns(UPDATE_TYPE);

            DataStream<IterationState> newState = updates
                .map((MapFunction<IterationUpdate, IterationState>) update -> update.state)
                .returns(STATE_TYPE)
                .setParallelism(1);

            DataStream<Integer> terminationSignal = updates
                .flatMap(new TerminationEvaluator())
                .returns(Types.INT)
                .setParallelism(1);

            return new IterationBodyResult(
                DataStreamList.of(newState),
                DataStreamList.of(newState),
                terminationSignal);
        }
    }

    /**
     * Parallel per-subtask local aggregation.
     * Computes partial distance sums for BUILD or SWAP phases based on iteration state.
     */
    private static final class PartialAggregator
            extends AbstractStreamOperator<PartialStats>
            implements TwoInputStreamOperator<WeightedPoint, IterationState, PartialStats>,
                       IterationListener<PartialStats> {

        private final double[][] candidateCoords;
        private final int targetK;
        private final int numCandidates;
        private final DistanceMetric distanceMetric;
        private transient ListStateWithCache<WeightedPoint> cachedPoints;
        private transient ListState<IterationState> stateList;

        PartialAggregator(double[][] candidateCoords, int targetK, DistanceMetric distanceMetric) {
            this.candidateCoords = candidateCoords;
            this.targetK = targetK;
            this.numCandidates = candidateCoords.length;
            this.distanceMetric = distanceMetric;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            stateList = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("dfastpam-state", STATE_TYPE));
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
            stateList.add(record.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<PartialStats> out) throws Exception {
            Optional<IterationState> currentStateOpt = OperatorStateUtils.getUniqueElement(stateList, "dfastpam-state");
            if (!currentStateOpt.isPresent()) {
                return;
            }
            IterationState state = currentStateOpt.get();
            int subtaskId = getRuntimeContext().getIndexOfThisSubtask();

            double[] partialData = state.numSelected < targetK ? aggregateBuildPhase(state) : aggregateSwapPhase(state);

            PartialStats stats = new PartialStats();
            stats.subtaskId = subtaskId;
            stats.accumulatedData = partialData;
            stats.state = state;
            out.collect(stats);

            stateList.clear();
        }

        private double[] aggregateBuildPhase(IterationState state) throws Exception {
            double[] accumulator = new double[numCandidates];
            if (state.numSelected == 0) {
                for (WeightedPoint point : cachedPoints.get()) {
                    double[] coords = point.features.values;
                    double weight = point.weight;
                    for (int i = 0; i < numCandidates; i++) {
                        accumulator[i] += weight * distanceMetric.compute(candidateCoords[i], coords);
                    }
                }
            } else {
                double[][] selectedMedoids = extractSelectedCoordinates(candidateCoords, state.medoids, state.numSelected);
                for (WeightedPoint point : cachedPoints.get()) {
                    double[] coords = point.features.values;
                    double weight = point.weight;
                    double minDistance = Double.MAX_VALUE;

                    for (double[] medoid : selectedMedoids) {
                        double dist = distanceMetric.compute(medoid, coords);
                        if (dist < minDistance) {
                            minDistance = dist;
                        }
                    }
                    for (int i = 0; i < numCandidates; i++) {
                        double gain = minDistance - distanceMetric.compute(candidateCoords[i], coords);
                        if (gain > 0.0) {
                            accumulator[i] += weight * gain;
                        }
                    }
                }
            }
            return accumulator;
        }

        private double[] aggregateSwapPhase(IterationState state) throws Exception {
            double[][] activeMedoids = extractSelectedCoordinates(candidateCoords, state.medoids, targetK);
            double[] accumulator = new double[numCandidates + numCandidates * targetK];

            for (WeightedPoint point : cachedPoints.get()) {
                double[] coords = point.features.values;
                double weight = point.weight;
                double closestDist = Double.MAX_VALUE;
                double secondClosestDist = Double.MAX_VALUE;
                int closestMedoidIdx = -1;

                for (int m = 0; m < targetK; m++) {
                    double dist = distanceMetric.compute(activeMedoids[m], coords);
                    if (dist < closestDist) {
                        secondClosestDist = closestDist;
                        closestDist = dist;
                        closestMedoidIdx = m;
                    } else if (dist < secondClosestDist) {
                        secondClosestDist = dist;
                    }
                }

                for (int i = 0; i < numCandidates; i++) {
                    double candidateDist = distanceMetric.compute(candidateCoords[i], coords);
                    double sharedContribution = candidateDist < closestDist ? weight * (candidateDist - closestDist) : 0.0;
                    accumulator[i] += sharedContribution;

                    double removeLoss = weight * (Math.min(secondClosestDist, candidateDist) - closestDist);
                    accumulator[numCandidates + i * targetK + closestMedoidIdx] += (removeLoss - sharedContribution);
                }
            }
            return accumulator;
        }

        @Override
        public void onIterationTerminated(Context context, Collector<PartialStats> out) throws Exception {
            cachedPoints.clear();
        }

        @Override public void processWatermark1(Watermark mark) {}
        @Override public void processWatermark2(Watermark mark) {}
        @Override public void processLatencyMarker1(LatencyMarker latencyMarker) {}
        @Override public void processLatencyMarker2(LatencyMarker latencyMarker) {}
    }

    /**
     * Single-task combiner. Merges partial results and determines the next BUILD selection
     * or the best SWAP move.
     */
    private static final class StatsCombiner
            implements FlatMapFunction<PartialStats, IterationUpdate>, IterationListener<IterationUpdate> {

        private final int numCandidates;
        private final int targetK;
        private final int maxIterations;
        private transient List<PartialStats> statsBuffer;

        StatsCombiner(int numCandidates, int targetK, int maxIterations) {
            this.numCandidates = numCandidates;
            this.targetK = targetK;
            this.maxIterations = maxIterations;
        }

        @Override
        public void flatMap(PartialStats stats, Collector<IterationUpdate> out) {
            if (statsBuffer == null) {
                statsBuffer = new ArrayList<>();
            }
            statsBuffer.add(stats);
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<IterationUpdate> out) {
            if (statsBuffer == null || statsBuffer.isEmpty()) {
                return;
            }
            statsBuffer.sort(Comparator.comparingInt(p -> p.subtaskId));
            IterationState state = statsBuffer.get(0).state;

            int dataLength = statsBuffer.get(0).accumulatedData.length;
            double[] globalSums = new double[dataLength];
            for (PartialStats partial : statsBuffer) {
                for (int i = 0; i < dataLength; i++) {
                    globalSums[i] += partial.accumulatedData[i];
                }
            }
            statsBuffer = null;

            IterationState nextState = state.copy();
            boolean shouldStop;

            if (state.numSelected < targetK) {
                boolean[] excluded = new boolean[numCandidates];
                for (int i = 0; i < state.numSelected; i++) {
                    excluded[state.medoids[i]] = true;
                }
                int nextIndex = (state.numSelected == 0)
                    ? findMinExcluding(globalSums, excluded)
                    : findMaxExcluding(globalSums, excluded);

                nextState.medoids[state.numSelected] = nextIndex;
                nextState.numSelected = state.numSelected + 1;
                shouldStop = false;
            } else {
                boolean[] isMedoid = new boolean[numCandidates];
                for (int i = 0; i < targetK; i++) {
                    isMedoid[state.medoids[i]] = true;
                }

                SwapMove bestMove = SwapMove.NONE;
                for (int candidateIdx = 0; candidateIdx < numCandidates; candidateIdx++) {
                    if (isMedoid[candidateIdx]) {
                        continue;
                    }
                    double sharedGain = globalSums[candidateIdx];
                    int baseIndex = numCandidates + candidateIdx * targetK;

                    for (int slot = 0; slot < targetK; slot++) {
                        double delta = sharedGain + globalSums[baseIndex + slot];
                        if (delta < 0.0) {
                            bestMove = SwapMove.getPreferredMove(bestMove, new SwapMove(delta, slot, candidateIdx));
                        }
                    }
                }
                nextState.completedSwapRounds = state.completedSwapRounds + 1;
                boolean isImprovement = SwapMove.isImprovement(bestMove);

                if (isImprovement) {
                    nextState.medoids[bestMove.targetSlot] = bestMove.candidateIdx;
                }
                shouldStop = !isImprovement || nextState.completedSwapRounds >= maxIterations;
            }

            IterationUpdate update = new IterationUpdate();
            update.state = nextState;
            update.shouldStop = shouldStop;
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
        public int[] medoids;
        public int numSelected;
        public int completedSwapRounds;

        public IterationState() {}

        static IterationState initialize(int targetK) {
            IterationState state = new IterationState();
            state.medoids = new int[targetK];
            Arrays.fill(state.medoids, -1);
            state.numSelected = 0;
            state.completedSwapRounds = 0;
            return state;
        }

        IterationState copy() {
            IterationState copy = new IterationState();
            copy.medoids = medoids.clone();
            copy.numSelected = numSelected;
            copy.completedSwapRounds = completedSwapRounds;
            return copy;
        }
    }

    public static final class PartialStats implements Serializable {
        public int subtaskId;
        public double[] accumulatedData;
        public IterationState state;

        public PartialStats() {}
    }

    public static final class IterationUpdate implements Serializable {
        public IterationState state;
        public boolean shouldStop;

        public IterationUpdate() {}
    }

    private static double[][] extractSelectedCoordinates(double[][] candidates, int[] indices, int count) {
        double[][] selected = new double[count][];
        for (int i = 0; i < count; i++) {
            selected[i] = candidates[indices[i]];
        }
        return selected;
    }

    private static int[] generateInitialIndices(int limit, int count) {
        int[] indices = new int[count];
        for (int i = 0; i < count; i++) {
            indices[i] = i;
        }
        return indices;
    }

    private static int findMinExcluding(double[] values, boolean[] excluded) {
        int bestIdx = -1;
        double minValue = Double.MAX_VALUE;
        for (int i = 0; i < values.length; i++) {
            if (!excluded[i] && values[i] < minValue) {
                minValue = values[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    private static int findMaxExcluding(double[] values, boolean[] excluded) {
        int bestIdx = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            if (!excluded[i] && values[i] > maxValue) {
                maxValue = values[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
}
