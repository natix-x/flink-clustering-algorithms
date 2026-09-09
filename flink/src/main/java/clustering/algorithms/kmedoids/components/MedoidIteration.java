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

/** The one FLIP-176 bounded iteration every distributed k-medoids entry runs in.
 *
 *  <p><b>Why a scaffold and not three iteration bodies.</b> {@code clara}, {@code pamae} and
 *  {@code distfastpam} all want the identical Flink shape — read the source ONCE, cache the points
 *  per subtask, then drive an arbitrary number of rounds over that cache from a single-task driver
 *  — and differ only in what a round computes. Writing that shape three times cost ~400 lines of
 *  duplicated operator boilerplate and, more importantly, made "one algorithm = one job" a property
 *  each class had to re-earn: {@code pamae} silently ran THREE jobs (CLARA, candidate pool,
 *  refinement), i.e. three reads of storage and three job deployments, because its phases lived in
 *  three places. Here a phase is a round, so adding one costs a branch, not a job.
 *
 *  <p>The cost of a Flink job is why this matters more here than on Spark: a job deployment
 *  measured 4 934 ms of fixed cost on Ares against Spark's 1 314 ms, and — since Flink has no
 *  cross-job cache — every extra job also re-reads the source from storage. Spark's PAMAE pays
 *  neither (its jobs are cheap and its {@code persist} survives them), which is exactly the kind of
 *  engine asymmetry hard rule 2 says to implement around rather than mirror.
 *
 *  <h3>Round protocol</h3>
 *  <pre>
 *    round r:  every subtask   computePartial(r, state, itsCachedPoints)  -&gt; P
 *              [optional]      merge P's of a contiguous subtask GROUP    -&gt; P
 *              driver (p=1)    combine(r, state, partials in slot order)  -&gt; next state | stop
 *  </pre>
 *
 *  <p><b>The state does not ride back with the partials.</b> Each of the three old bodies attached
 *  the whole {@code IterationState} to every subtask's {@code PartialStats} so the combiner could
 *  read it — sending CLARA's candidate sets (numSamples·k·d doubles, 400 KB at k=10, d=1024) from
 *  all 64 subtasks into ONE task, every round, to re-learn something that task had emitted itself
 *  one round earlier. The combiner keeps its own copy instead, so a partial carries only its slot
 *  and its numbers.
 *
 *  <h3>Ordered merging</h3>
 *  Partials are folded in ascending {@code slot} order, never in arrival order, because these folds
 *  ARE the result: k-medoids picks an argmin over sums of doubles, so a completion-ordered merge
 *  makes two runs of one configuration disagree in the last bits and, occasionally, in the medoid.
 *  This is the counterpart of the Spark side's {@code PartitionAggregator.aggregateDoublesOrdered},
 *  and it was learned the same way there.
 *
 *  <p>{@code mergeFanIn > 1} inserts an intermediate stage that folds each contiguous group of
 *  {@code mergeFanIn} subtasks before the driver sees them, cutting what arrives at the single
 *  final task by that factor. It exists for {@code distfastpam}, whose partial is
 *  {@code n + n·k} doubles — 880 KB per subtask at n = 10 000, k = 10, i.e. 56 MB per round through
 *  one task at parallelism 64, every round. The grouping is by subtask RANGE
 *  ({@code slot / mergeFanIn}) and the partitioner sends a whole group to one task, so the fold
 *  order is still fully determined by slot indices and the tree changes nothing but the wire
 *  volume. */
public final class MedoidIteration {

    private MedoidIteration() {}

    /** Base of every iteration state.
     *
     *  <p>{@link #stop} rides ALONG the state rather than in a wrapper record because Flink has to
     *  serialize whatever crosses the feedback edge, and a wrapper holding the state behind an
     *  {@code Object} field drops the whole graph to Kryo — which, on a JDK 17 build, does not even
     *  initialise ({@code java.base does not "opens java.util"}). Every state here is a strict POJO
     *  instead, so Flink serializes it field by field with no reflection into the JDK. */
    public abstract static class State implements Serializable {
        /** Set by the framework from the round's {@link Decision}; read by the termination stream. */
        public boolean stop;
    }

    /** Base of every per-subtask partial result.
     *
     *  <p>{@link #slot} is the subtask index as emitted, and the GROUP index after a merge stage.
     *  It is the sort key of both folds, so it is what makes the run reproducible; the framework
     *  sets it and a {@link RoundLogic} never should. */
    public abstract static class Partial implements Serializable {
        public int slot;
    }

    /** What the driver decided after folding one round's partials. */
    public static final class Decision<S> {
        public final S nextState;
        public final boolean stop;

        private Decision(S nextState, boolean stop) {
            this.nextState = nextState;
            this.stop = stop;
        }

        /** Run another round with {@code nextState}. */
        public static <S> Decision<S> next(S nextState) {
            return new Decision<>(nextState, false);
        }

        /** Stop; {@code finalState} is what {@link #execute} returns. */
        public static <S> Decision<S> stop(S finalState) {
            return new Decision<>(finalState, true);
        }
    }

    /** The algorithm-specific half of a round: what a subtask computes, and what the driver
     *  concludes. Instances are deserialized ONCE per subtask and reused across rounds, so a
     *  {@code transient} field is a legitimate per-subtask memo — which is how a payload that must
     *  reach every worker (a candidate pool, the full candidate set) is broadcast on ONE round and
     *  then omitted from the state for the rest of the run. */
    public interface RoundLogic<S extends State, P extends Partial> extends Serializable {

        /** This subtask's contribution for round {@code round}. Called once per round, after every
         *  point has reached the cache; {@code points} is that cache and may be empty. */
        P computePartial(int round, S state, Iterable<WeightedPoint> points, int subtaskId) throws Exception;

        /** Associative fold of two partials of the SAME round, {@code left.slot < right.slot}.
         *  Only needed when {@code mergeFanIn > 1}. */
        default P merge(P left, P right) {
            throw new UnsupportedOperationException(
                getClass().getName() + " has no merge(); do not pass mergeFanIn > 1");
        }

        /** Driver-side (parallelism 1) fold of a round's partials, in ascending slot order. */
        Decision<S> combine(int round, S state, List<P> partials);
    }

    /** Runs the iteration as ONE Flink job and returns the state the last round decided on. */
    public static <S extends State, P extends Partial> S execute(
            PointSource source,
            EnvFactory envFactory,
            String jobName,
            S initialState,
            Class<S> stateClass,
            Class<P> partialClass,
            RoundLogic<S, P> logic,
            int mergeFanIn) {

        TypeInformation<S> stateType = TypeInformation.of(stateClass);
        TypeInformation<P> partialType = TypeInformation.of(partialClass);

        StreamExecutionEnvironment env = envFactory.newEnv();
        // FLIP-176 iterations require STREAMING; the AdaptiveBatchScheduler does not apply here,
        // which is also why the source keeps the configured parallelism inside an iteration.
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);

        DataStream<S> initStateStream =
            env.fromCollection(Collections.singletonList(initialState), stateType);
        DataStream<WeightedPoint> pointsStream = source.create(env);

        DataStreamList result = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initStateStream),
            ReplayableDataStreamList.notReplay(pointsStream),
            IterationConfig.newBuilder().build(),
            new Body<>(logic, initialState, stateType, partialType, mergeFanIn));

        return FlinkJobs.last(result.<S>get(0), jobName);
    }

    private static final class Body<S extends State, P extends Partial> implements IterationBody {

        private final RoundLogic<S, P> logic;
        private final S initialState;
        private final TypeInformation<S> stateType;
        private final TypeInformation<P> partialType;
        private final int mergeFanIn;

        Body(RoundLogic<S, P> logic, S initialState, TypeInformation<S> stateType,
             TypeInformation<P> partialType, int mergeFanIn) {
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

            // Without this the ListStateWithCache below gets a null memory-segment pool and every
            // cached point goes straight to io.tmp.dirs — see ManagedMemory.
            ManagedMemory.forPointCache(partials);

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
                terminationSignal);
        }
    }

    /** Sends a whole contiguous subtask group to one merge task, so a group is never split and the
     *  fold order stays a function of slot indices alone. */
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

    /** Caches this subtask's points once and answers one {@link RoundLogic#computePartial} per
     *  round. The cache is a {@code ListStateWithCache} (managed memory, spilling to disk) — the
     *  Flink counterpart of Spark's {@code persist(MEMORY_AND_DISK)} on the input. */
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

    /** Optional pre-fold: one merged partial per contiguous subtask group, folded in slot order. */
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

    /** The driver: folds a round's partials in slot order and decides what happens next. It keeps
     *  the state itself rather than reading it back off the partials — see the class doc. */
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
