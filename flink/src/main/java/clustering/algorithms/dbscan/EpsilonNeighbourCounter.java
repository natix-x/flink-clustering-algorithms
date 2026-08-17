package clustering.algorithms.dbscan;

import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
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

/** For every candidate point, the number of dataset points within eps of it.
 *
 *  Shared by {@code dbscanpp} (core-point test {@code density >= minPts}) and, once it exists,
 *  {@code dpc} (local density rho). Counted against ALL n points, so densities are EXACT; only
 *  the candidate set is sampled. Cost: n x m distance computations.
 *
 *  <h3>One job, one pass — the difference from the Spark implementation</h3>
 *  Spark ships the candidates in broadcast chunks and runs ONE JOB PER CHUNK, i.e. one full
 *  pass over the data per chunk (m/chunkSize passes over 26 GB of Gaia, and the same again for
 *  the next run). Here the chunks are ROUNDS of a single FLIP-176 job: the points are cached
 *  per subtask once ({@link ListStateWithCache}, memory + disk spill) and only the current
 *  chunk of candidate coordinates travels the feedback edge, so chunking still bounds the
 *  per-subtask memory (its whole purpose) while the data is read exactly once.
 *
 *  {@code chunkSize} therefore keeps its meaning as "candidate coordinates resident per
 *  subtask" and stays comparable with the Spark knob; what it no longer costs is a re-read.
 *
 *  <h3>Weights</h3>
 *  The Flink seam has no weight channel, so counts are plain row counts and {@code minPts} is a
 *  threshold on rows — identical to a Spark run whose input has no {@code weight} column. */
public final class EpsilonNeighbourCounter {

    private static final Logger logger = LoggerFactory.getLogger(EpsilonNeighbourCounter.class);

    private static final TypeInformation<State> STATE_TYPE = TypeInformation.of(State.class);
    private static final TypeInformation<Partial> PARTIAL_TYPE = TypeInformation.of(Partial.class);
    private static final TypeInformation<Update> UPDATE_TYPE = TypeInformation.of(Update.class);
    private static final TypeInformation<ChunkCounts> CHUNK_COUNTS_TYPE = TypeInformation.of(ChunkCounts.class);

    /** Hard ceiling on the candidates counted per round.
     *
     *  A round's counts leave the job as ONE record, and Flink's collect sink refuses a record
     *  above {@code collect-sink.batch-size.max} (2 MB by default) — which is what a 2·10^6-wide
     *  count array hit after half an hour of correct counting. A chunk of 100 000 makes that
     *  record 800 KB, comfortably inside the default, and 100 000 candidates per pass is already
     *  far past the point where the pass cost is dominated by the scan rather than by the round. */
    private static final int MaxChunkSize = 100_000;

    private EpsilonNeighbourCounter() {}

    /** {@code result[i]} = number of dataset points x with {@code d(candidates[i], x) <= eps}.
     *
     *  A candidate that is itself a dataset point counts itself (the DBSCAN core condition). */
    public static double[] computeNeighbourhoodDensities(
            PointSource source,
            EnvFactory envs,
            double[][] candidates,
            double eps,
            DistanceMetric distanceMetric,
            int chunkSize) {
        if (candidates.length == 0) {
            return new double[0];
        }
        int chunk = Math.max(1, Math.min(chunkSize, candidates.length));
        if (chunk > MaxChunkSize) {
            logger.warn("epsilon-counting: chunkSize {} capped to {} — a round's counts travel as one "
                + "record and the collect sink refuses anything above ~2 MB", chunk, MaxChunkSize);
            chunk = MaxChunkSize;
        }

        // FLIP-176 iterations add feedback edges (unbounded graph) -> STREAMING mode.
        StreamExecutionEnvironment env = envs.newEnv();
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);

        DataStream<State> initState =
            env.fromCollection(Collections.singletonList(State.firstChunk(candidates, chunk)), STATE_TYPE);
        DataStream<DenseVector> points = source.create(env);

        DataStreamList result = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initState),
            ReplayableDataStreamList.notReplay(points),
            IterationConfig.newBuilder().build(),
            new CountingIterationBody(candidates, chunk, eps, distanceMetric));

        // The counts come back CHUNK BY CHUNK and are folded into the result as they arrive, for
        // the same reason the eps-graph streams its edges: one record per finished chunk stays
        // small, whereas the whole m-wide array is a single record the collect sink would reject
        // (2·10^6 candidates = 16 MB against a 2 MB default limit).
        double[] counts = new double[candidates.length];
        boolean[] filled = new boolean[candidates.length];
        FlinkJobs.consume(result.<ChunkCounts>get(0), "epsilon-counting", record -> {
            System.arraycopy(record.counts, 0, counts, record.chunkStart, record.counts.length);
            for (int i = 0; i < record.counts.length; i++) {
                filled[record.chunkStart + i] = true;
            }
        });
        for (int i = 0; i < filled.length; i++) {
            if (!filled[i]) {
                throw new RuntimeException("epsilon-counting produced no density for candidate " + i
                    + " of " + candidates.length + " — a chunk never came back");
            }
        }
        return counts;
    }

    /** One round = one candidate chunk: per-subtask counting (parallel) -> accumulate (1). */
    private static final class CountingIterationBody implements IterationBody {
        private final double[][] candidates;
        private final int chunkSize;
        private final double eps;
        private final DistanceMetric distance;

        CountingIterationBody(double[][] candidates, int chunkSize, double eps, DistanceMetric distance) {
            this.candidates = candidates;
            this.chunkSize = chunkSize;
            this.eps = eps;
            this.distance = distance;
        }

        @Override
        public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<State> state = variableStreams.get(0);
            DataStream<DenseVector> points = dataStreams.get(0);

            DataStream<Partial> partials = points
                .connect(state.broadcast())
                .transform("epsilon-count-chunk", PARTIAL_TYPE, new CountFold(eps, distance));

            DataStream<Update> updates = partials
                .flatMap(new Accumulate(candidates, chunkSize))
                .setParallelism(1)
                .returns(UPDATE_TYPE);

            DataStream<State> newState = updates
                .map((MapFunction<Update, State>) u -> u.state)
                .returns(STATE_TYPE)
                .setParallelism(1);

            DataStream<Integer> termination = updates
                .flatMap(new ContinueUnlessStop())
                .returns(Types.INT)
                .setParallelism(1);

            // The iteration's OUTPUT is one small record per finished chunk, not the final state:
            // that is what keeps a record's size tied to chunkSize instead of to m.
            DataStream<ChunkCounts> chunkCounts = updates
                .map((MapFunction<Update, ChunkCounts>) u -> u.chunkCounts)
                .returns(CHUNK_COUNTS_TYPE)
                .setParallelism(1);

            return new IterationBodyResult(
                DataStreamList.of(newState),
                DataStreamList.of(chunkCounts),
                termination);
        }
    }

    /** Parallel per-subtask counting fold: for the round's candidate chunk, how many of THIS
     *  subtask's points fall within eps of each candidate. */
    private static final class CountFold
            extends AbstractStreamOperator<Partial>
            implements TwoInputStreamOperator<DenseVector, State, Partial>,
                       IterationListener<Partial> {

        private final double eps;
        private final DistanceMetric distance;
        private transient ListStateWithCache<double[]> points;
        private transient ListState<State> stateList;

        CountFold(double eps, DistanceMetric distance) {
            this.eps = eps;
            this.distance = distance;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            stateList = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("epsilon-count-state", STATE_TYPE));
            points = new ListStateWithCache<>(
                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO.createSerializer(getExecutionConfig()),
                getContainingTask(),
                getRuntimeContext(),
                context,
                config.getOperatorID());
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            points.snapshotState(context);
        }

        @Override
        public void processElement1(StreamRecord<DenseVector> record) throws Exception {
            // Unwrap at the boundary: cache, state serializer and the scan all use the raw array.
            points.add(record.getValue().values);
        }

        @Override
        public void processElement2(StreamRecord<State> record) throws Exception {
            stateList.add(record.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<Partial> out) throws Exception {
            Optional<State> current = OperatorStateUtils.getUniqueElement(stateList, "epsilon-count-state");
            if (!current.isPresent()) {
                return;
            }
            State s = current.get();
            double[][] chunk = s.chunk;
            double[] acc = new double[chunk.length];
            for (double[] x : points.get()) {
                for (int i = 0; i < chunk.length; i++) {
                    if (distance.withinRadius(chunk[i], x, eps)) {
                        acc[i] += 1.0;
                    }
                }
            }
            Partial p = new Partial();
            p.subtask = getRuntimeContext().getIndexOfThisSubtask();
            p.data = acc;
            p.chunkStart = s.chunkStart;
            out.collect(p);
            stateList.clear();
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Partial> out) throws Exception {
            points.clear();
        }

        @Override public void processWatermark1(Watermark mark) {}
        @Override public void processWatermark2(Watermark mark) {}
        @Override public void processLatencyMarker1(LatencyMarker latencyMarker) {}
        @Override public void processLatencyMarker2(LatencyMarker latencyMarker) {}
    }

    /** Single-task combiner: sums the round's partials (subtask order, so the sum is
     *  deterministic), EMITS that chunk's finished counts, and ships the next candidate chunk.
     *
     *  It deliberately keeps no m-wide array: the driver owns the assembled result, the combiner
     *  only ever holds one chunk. That is what bounds the emitted record — and this task's memory —
     *  by {@code chunkSize} rather than by m. */
    private static final class Accumulate
            implements FlatMapFunction<Partial, Update>, IterationListener<Update> {

        private final double[][] candidates;
        private final int chunkSize;
        private transient List<Partial> buffer;
        private transient long startedAtNanos;
        private transient int chunksDone;

        Accumulate(double[][] candidates, int chunkSize) {
            this.candidates = candidates;
            this.chunkSize = chunkSize;
        }

        @Override
        public void flatMap(Partial partial, Collector<Update> out) {
            if (buffer == null) {
                buffer = new ArrayList<>();
            }
            buffer.add(partial);
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<Update> out) {
            if (buffer == null || buffer.isEmpty()) {
                return;
            }
            if (chunksDone == 0) {
                startedAtNanos = System.nanoTime();
            }
            buffer.sort(Comparator.comparingInt(p -> p.subtask));
            int chunkStart = buffer.get(0).chunkStart;
            int len = buffer.get(0).data.length;
            double[] chunkCounts = new double[len];
            for (Partial p : buffer) {
                for (int i = 0; i < len; i++) {
                    chunkCounts[i] += p.data[i];
                }
            }
            buffer = null;

            int nextStart = chunkStart + len;
            chunksDone++;
            int chunkCount = (candidates.length + chunkSize - 1) / chunkSize;
            double elapsed = (System.nanoTime() - startedAtNanos) / 1e9;
            logger.info(String.format("epsilon-counting: chunk %d/%d (%d candidates), %.0f s elapsed, %.0f s left",
                chunksDone, chunkCount, len, elapsed, elapsed * (chunkCount - chunksDone) / chunksDone));

            Update u = new Update();
            u.chunkCounts = ChunkCounts.of(chunkStart, chunkCounts);
            if (nextStart >= candidates.length) {
                // Nothing left to count: this state is never read again, it only has to be a
                // well-formed record for the feedback edge.
                u.state = State.chunkAt(candidates, chunkStart, len);
                u.stop = true;
            } else {
                u.state = State.chunkAt(candidates, nextStart, chunkSize);
                u.stop = false;
            }
            out.collect(u);
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Update> out) {}
    }

    /** Emits a "continue" token each round unless the round's {@link Update} says stop. */
    private static final class ContinueUnlessStop
            implements FlatMapFunction<Update, Integer>, IterationListener<Integer> {
        private transient boolean stop;

        @Override
        public void flatMap(Update update, Collector<Integer> out) {
            stop = update.stop;
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

    // ── POJOs cycled through the iteration ────────────────────────────────────

    /** Feedback state: the candidate chunk to count in the coming round. Carries no counts —
     *  those leave the job on the output stream, one small record per chunk. */
    public static final class State implements Serializable {
        public int chunkStart;
        public double[][] chunk;

        public State() {}

        static State firstChunk(double[][] candidates, int chunkSize) {
            return chunkAt(candidates, 0, chunkSize);
        }

        static State chunkAt(double[][] candidates, int start, int chunkSize) {
            State s = new State();
            s.chunkStart = start;
            s.chunk = Arrays.copyOfRange(candidates, start, Math.min(candidates.length, start + chunkSize));
            return s;
        }
    }

    /** Per-subtask counts for one chunk. */
    public static final class Partial implements Serializable {
        public int subtask;
        public int chunkStart;
        public double[] data;

        public Partial() {}
    }

    /** Combined per-round result: the finished chunk's counts (the job's output), the next
     *  {@link State} (the feedback edge) + whether to stop. */
    public static final class Update implements Serializable {
        public ChunkCounts chunkCounts;
        public State state;
        public boolean stop;

        public Update() {}
    }

    /** One finished chunk's counts, positioned by {@code chunkStart} — the record the driver folds
     *  into the result array. Sized by {@code chunkSize}, never by m: see {@link #MaxChunkSize}. */
    public static final class ChunkCounts implements Serializable {
        public int chunkStart;
        public double[] counts;

        public ChunkCounts() {}

        static ChunkCounts of(int chunkStart, double[] counts) {
            ChunkCounts c = new ChunkCounts();
            c.chunkStart = chunkStart;
            c.counts = counts;
            return c;
        }
    }
}
