package clustering.algorithms.kmedoids.components;

import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.ManagedMemory;
import clustering.core.PointSource;
import clustering.core.WeightedPoint;
import clustering.core.WeightedPointTypeInfo;
import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.Partitioner;
import org.apache.flink.api.java.functions.KeySelector;
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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * Scaffold for running bounded FLIP-176 distributed k-medoids iterations in Flink.
 * Executes a compute-merge-combine protocol: reads the source once, caches points per subtask,
 * and drives an arbitrary number of rounds over that cache.
 * Partials are merged strictly in ascending slot order to guarantee deterministic results.
 */
public final class MedoidIteration {

    private MedoidIteration() {}

    /**
     * Base for every iteration state.
     * Must remain a strict POJO to ensure correct Flink field-by-field serialization.
     */
    public abstract static class State implements Serializable {
        public boolean stop;
    }

    /**
     * Base for every per-subtask partial result.
     * The {@code slot} is used as the sort key during merging to guarantee reproducible runs.
     */
    public abstract static class Partial implements Serializable {
        public int slot;
    }

    /**
     * Driver's decision after folding a round's partials.
     */
    public static final class Decision<S> {
        public final S nextState;
        public final boolean stop;

        private Decision(S nextState, boolean stop) {
            this.nextState = nextState;
            this.stop = stop;
        }

        public static <S> Decision<S> next(S nextState) {
            return new Decision<>(nextState, false);
        }

        public static <S> Decision<S> stop(S finalState) {
            return new Decision<>(finalState, true);
        }
    }

    /**
     * The algorithm-specific logic for a single round.
     * Defines what each subtask computes and how the driver combines those partials.
     */
    public interface RoundLogic<S extends State, P extends Partial> extends Serializable {

        P computePartial(int round, S state, Iterable<WeightedPoint> points, int subtaskId) throws Exception;

        default P merge(P left, P right) {
            throw new UnsupportedOperationException(
                getClass().getName() + " has no merge(); do not pass mergeFanIn > 1");
        }

        Decision<S> combine(int round, S state, List<P> partials);
    }

    /**
     * Runs the iteration as a single Flink job and returns the final decided state.
     */
    public static <S extends State, P extends Partial> S execute(
            PointSource source,
            EnvFactory envFactory,
            String jobName,
            S initialState,
            Class<S> stateClass,
            Class<P> partialClass,
            RoundLogic<S, P> logic,
            int mergeFanIn
    ) {
        TypeInformation<S> stateType = TypeInformation.of(stateClass);
        TypeInformation<P> partialType = TypeInformation.of(partialClass);

        StreamExecutionEnvironment env = envFactory.newEnv();
        // FLIP-176 requires STREAMING mode.
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);

        DataStream<S> initStateStream = env.fromCollection(Collections.singletonList(initialState), stateType);
        DataStream<WeightedPoint> pointsStream = source.create(env);

        DataStreamList result = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initStateStream),
            ReplayableDataStreamList.notReplay(pointsStream),
            IterationConfig.newBuilder().build(),
            new Body<>(logic, initialState, stateType, partialType, mergeFanIn)
        );

        return FlinkJobs.last(result.<S>get(0), jobName);
    }

    private static final class Body<S extends State, P extends Partial> implements IterationBody {

        private final RoundLogic<S, P> logic;
        private final S initialState;
        private final TypeInformation<S> stateType;
        private final TypeInformation<P> partialType;
        private final int mergeFanIn;

        Body(
            RoundLogic<S, P> logic,
            S initialState,
            TypeInformation<S> stateType,
            TypeInformation<P> partialType,
            int mergeFanIn
        ) {
            this.logic = logic;
            this.initialState = initialState;
            this.stateType = stateType;
            this.partialType = partialType;
            this.mergeFanIn = mergeFanIn;
        }

        @Override
        public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<S> stateStream = variableStreams.get(0);
            DataStream<WeightedPoint> pointsStream = dataStreams.get(0);

            DataStream<P> partials = pointsStream
                .connect(stateStream.broadcast())
                .transform("medoid-round", partialType, new RoundOperator<>(logic, stateType));

            // Required to initialize managed memory for the point cache to avoid spilling directly to io.tmp.dirs.
            ManagedMemory.allocateForPointCache(partials);

            DataStream<P> folded = partials;
            if (mergeFanIn > 1) {
                folded = partials
                    .partitionCustom(new GroupPartitioner(mergeFanIn), (KeySelector<P, Integer>) p -> p.slot)
                    .flatMap(new MergeStage<>(logic, mergeFanIn))
                    .returns(partialType);
            }

            DataStream<S> newState = folded
                .flatMap(new DecisionCombiner<>(logic, initialState))
                .setParallelism(1)
                .returns(stateType);

            DataStream<Integer> terminationSignal = newState
                .flatMap(new TerminationEvaluator<S>())
                .returns(Types.INT)
                .setParallelism(1);

            return new IterationBodyResult(
                DataStreamList.of(newState),
                DataStreamList.of(newState),
                terminationSignal
            );
        }
    }

    /**
     * Partitions contiguous subtask groups to a single merge task to preserve slot-based fold order.
     */
    private static final class GroupPartitioner implements Partitioner<Integer> {
        private final int fanIn;

        GroupPartitioner(int fanIn) {
            this.fanIn = fanIn;
        }

        @Override
        public int partition(Integer slot, int numPartitions) {
            return (slot / fanIn) % numPartitions;
        }
    }

    /**
     * Caches subtask points once and executes one computePartial round per epoch.
     */
    private static final class RoundOperator<S extends State, P extends Partial>
            extends AbstractStreamOperator<P>
            implements TwoInputStreamOperator<WeightedPoint, S, P>, IterationListener<P> {

        private static final String STATE_NAME = "medoid-iteration-state";

        private final RoundLogic<S, P> logic;
        private final TypeInformation<S> stateType;
        private transient ListStateWithCache<WeightedPoint> cachedPoints;
        private transient ListState<S> stateList;

        RoundOperator(RoundLogic<S, P> logic, TypeInformation<S> stateType) {
            this.logic = logic;
            this.stateType = stateType;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            stateList = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>(STATE_NAME, stateType));
            cachedPoints = new ListStateWithCache<>(
                WeightedPointTypeInfo.INSTANCE.createSerializer(getExecutionConfig()),
                getContainingTask(),
                getRuntimeContext(),
                context,
                config.getOperatorID()
            );
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
        public void processElement2(StreamRecord<S> record) throws Exception {
            stateList.add(record.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<P> out) throws Exception {
            Optional<S> state = OperatorStateUtils.getUniqueElement(stateList, STATE_NAME);
            if (!state.isPresent()) {
                return;
            }
            int subtaskId = getRuntimeContext().getIndexOfThisSubtask();
            P partial = logic.computePartial(epoch, state.get(), cachedPoints.get(), subtaskId);
            partial.slot = subtaskId;
            out.collect(partial);
            stateList.clear();
        }

        @Override
        public void onIterationTerminated(Context context, Collector<P> out) throws Exception {
            cachedPoints.clear();
        }

        @Override public void processWatermark1(Watermark mark) {}
        @Override public void processWatermark2(Watermark mark) {}
        @Override public void processLatencyMarker1(LatencyMarker latencyMarker) {}
        @Override public void processLatencyMarker2(LatencyMarker latencyMarker) {}
    }

    /**
     * Optional pre-fold stage merging partials within subtask groups.
     */
    private static final class MergeStage<S extends State, P extends Partial>
            implements FlatMapFunction<P, P>, IterationListener<P> {

        private final RoundLogic<S, P> logic;
        private final int fanIn;
        private transient List<P> buffer;

        MergeStage(RoundLogic<S, P> logic, int fanIn) {
            this.logic = logic;
            this.fanIn = fanIn;
        }

        @Override
        public void flatMap(P partial, Collector<P> out) {
            if (buffer == null) {
                buffer = new ArrayList<>();
            }
            buffer.add(partial);
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<P> out) {
            if (buffer == null || buffer.isEmpty()) {
                return;
            }
            buffer.sort(Comparator.comparingInt(p -> p.slot));

            int index = 0;
            while (index < buffer.size()) {
                int group = buffer.get(index).slot / fanIn;
                P merged = buffer.get(index);
                index++;
                while (index < buffer.size() && buffer.get(index).slot / fanIn == group) {
                    merged = logic.merge(merged, buffer.get(index));
                    index++;
                }
                merged.slot = group;
                out.collect(merged);
            }
            buffer = null;
        }

        @Override
        public void onIterationTerminated(Context context, Collector<P> out) {}
    }

    /**
     * Driver task that folds a round's partials in slot order and decides the next state.
     */
    private static final class DecisionCombiner<S extends State, P extends Partial>
            implements FlatMapFunction<P, S>, IterationListener<S> {

        private final RoundLogic<S, P> logic;
        private final S initialState;
        private transient S currentState;
        private transient List<P> buffer;

        DecisionCombiner(RoundLogic<S, P> logic, S initialState) {
            this.logic = logic;
            this.initialState = initialState;
        }

        @Override
        public void flatMap(P partial, Collector<S> out) {
            if (buffer == null) {
                buffer = new ArrayList<>();
            }
            buffer.add(partial);
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<S> out) {
            if (buffer == null || buffer.isEmpty()) {
                return;
            }
            if (currentState == null) {
                currentState = initialState;
            }
            buffer.sort(Comparator.comparingInt(p -> p.slot));

            Decision<S> decision = logic.combine(epoch, currentState, buffer);
            buffer = null;
            currentState = decision.nextState;
            currentState.stop = decision.stop;
            out.collect(currentState);
        }

        @Override
        public void onIterationTerminated(Context context, Collector<S> out) {}
    }

    private static final class TerminationEvaluator<S extends State>
            implements FlatMapFunction<S, Integer>, IterationListener<Integer> {

        private transient boolean stop;

        @Override
        public void flatMap(S state, Collector<Integer> out) {
            stop = state.stop;
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<Integer> out) {
            if (!stop) {
                out.collect(0);
            }
            stop = false;
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Integer> out) {}
    }
}
