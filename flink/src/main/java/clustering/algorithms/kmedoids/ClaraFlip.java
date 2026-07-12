package clustering.algorithms.kmedoids;

import clustering.core.Clusterer;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Random;

/** CLARA on the Flink ML bounded-iteration framework (FLIP-176) — the whole multi-sample
 *  search is ONE Flink job, mirroring {@link DistributedFastPAM}. This is the
 *  cluster-friendly CLARA: the full dataset is cached per subtask ONCE and reused for every
 *  sample's cost evaluation, instead of the plain {@link CLARA}'s {@code 2·numSamples}
 *  separate jobs (a sampling job + a full-dataset cost job per sample, re-reading the data
 *  each time).
 *
 *  <h3>Samples as iteration rounds</h3>
 *  Each round draws one deterministic sample from the cached points, ships it to the
 *  parallelism-1 combiner which runs driver-local {@link PAM#fitLocal} on it (the sample is
 *  small — that is the whole point of CLARA, so PAM stays local), and the resulting
 *  candidate medoids are fed back. The NEXT round folds that candidate's assignment cost
 *  over the FULL cached dataset (distributed), so the combiner can compare it to the
 *  best-so-far. After {@code numSamples} candidates have been generated and scored, the
 *  lowest-cost medoid set is emitted.
 *
 *  <h3>What is distributed vs local</h3>
 *  Distributed: caching the full dataset + the per-round full-dataset cost fold. Local (in
 *  the combiner, parallelism 1): PAM on the small sample. This keeps CLARA's design — scale
 *  by sampling, not by distributing PAM — while removing the per-sample job submission and
 *  repeated full-dataset reads. For LARGE samples that no longer fit the driver, use a
 *  distributed inner PAM instead (a separate variant).
 *
 *  <h3>Determinism</h3>
 *  Reproducible for a FIXED parallelism: samples are gathered and costs summed in
 *  subtask-id order, and {@link PAM#fitLocal} is deterministic. Across different parallelism
 *  the point partition changes, so the drawn sample (and thus the result) can differ. */
public class ClaraFlip implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(ClaraFlip.class);

    private static final TypeInformation<State> STATE_TYPE = TypeInformation.of(State.class);
    private static final TypeInformation<Partial> PARTIAL_TYPE = TypeInformation.of(Partial.class);
    private static final TypeInformation<Update> UPDATE_TYPE = TypeInformation.of(Update.class);

    private final int k;
    private final int numSamples;
    private final int sampleSize;
    private final int maxIter;
    private final DistanceMetric distance;

    public ClaraFlip(int k, int numSamples, int sampleSize, int maxIter, DistanceMetric distance) {
        this.k = k;
        this.numSamples = numSamples;
        this.sampleSize = sampleSize;
        this.maxIter = maxIter;
        this.distance = distance;
    }

    @Override
    public Model fit(PointSource source, EnvFactory envs, int parallelism) {
        long n = Datasets.count(source, envs);
        if (n < k) {
            throw new IllegalArgumentException(
                "Dataset too small: n=" + n + " points but k=" + k + " medoids requested.");
        }
        // Oversample 2x then cap at sampleSize in the combiner (mirrors plain CLARA).
        double fraction = Math.min(1.0, sampleSize * 2.0 / n);

        StreamExecutionEnvironment env = envs.newEnv();
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);

        DataStream<State> initState =
            env.fromCollection(java.util.Collections.singletonList(State.initial()), STATE_TYPE);
        DataStream<double[]> points = source.create(env);

        DataStreamList result = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initState),
            ReplayableDataStreamList.notReplay(points),
            IterationConfig.newBuilder().build(),
            new ClaraIterationBody(k, numSamples, sampleSize, maxIter, fraction, distance));

        State finalState = FlinkJobs.last(result.<State>get(0), "claraflip-fit");

        if (finalState == null || finalState.bestMedoids == null) {
            throw new RuntimeException("CLARA (FLIP-176) produced no medoids");
        }
        return new KMedoidsModel(finalState.bestMedoids, distance);
    }

    /** One round: per-subtask fold (sample + cost) -> combine (1: PAM on sample, pick best)
     *  -> feed back state; stop after numSamples candidates are scored. */
    private static final class ClaraIterationBody implements IterationBody {
        private final int k;
        private final int numSamples;
        private final int sampleSize;
        private final int maxIter;
        private final double fraction;
        private final DistanceMetric distance;

        ClaraIterationBody(int k, int numSamples, int sampleSize, int maxIter,
                           double fraction, DistanceMetric distance) {
            this.k = k;
            this.numSamples = numSamples;
            this.sampleSize = sampleSize;
            this.maxIter = maxIter;
            this.fraction = fraction;
            this.distance = distance;
        }

        @Override
        public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<State> state = variableStreams.get(0);
            DataStream<double[]> points = dataStreams.get(0);

            DataStream<Partial> partials = points
                .connect(state.broadcast())
                .transform("claraflip-fold", PARTIAL_TYPE, new SampleAndCostFold(fraction, sampleSize, distance));

            DataStream<Update> updates = partials
                .flatMap(new Combine(k, numSamples, sampleSize, maxIter, distance))
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

            return new IterationBodyResult(
                DataStreamList.of(newState),
                DataStreamList.of(newState),
                termination);
        }
    }

    /** Parallel per-subtask fold. Caches local points once. Each round it (a) draws a
     *  deterministic Bernoulli sample of its local points for THIS round, and (b) if the
     *  state carries a pending candidate, sums that candidate's nearest-medoid cost over
     *  ALL its local points. Emits one {@link Partial}. */
    private static final class SampleAndCostFold
            extends AbstractStreamOperator<Partial>
            implements TwoInputStreamOperator<double[], State, Partial>,
                       IterationListener<Partial> {

        private final double fraction;
        private final int sampleSize;
        private final DistanceMetric distance;
        private transient ListStateWithCache<double[]> points;
        private transient ListState<State> stateList;

        SampleAndCostFold(double fraction, int sampleSize, DistanceMetric distance) {
            this.fraction = fraction;
            this.sampleSize = sampleSize;
            this.distance = distance;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            stateList = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("claraflip-state", STATE_TYPE));
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
        public void processElement1(StreamRecord<double[]> record) throws Exception {
            points.add(record.getValue());
        }

        @Override
        public void processElement2(StreamRecord<State> record) throws Exception {
            stateList.add(record.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<Partial> out) throws Exception {
            Optional<State> current = OperatorStateUtils.getUniqueElement(stateList, "claraflip-state");
            if (!current.isPresent()) {
                return;
            }
            State s = current.get();
            int subtask = getRuntimeContext().getIndexOfThisSubtask();

            // (a) deterministic per-(round, subtask) Bernoulli sample of local points.
            Random rng = new Random(31L * (s.round + 1) + subtask);
            List<double[]> localSample = new ArrayList<>();
            // (b) cost of the pending candidate over ALL local points (0 if none yet).
            double cost = 0.0;
            boolean hasPending = s.pendingMedoids != null;
            for (double[] x : points.get()) {
                if (localSample.size() < sampleSize && rng.nextDouble() < fraction) {
                    localSample.add(x);
                }
                if (hasPending) {
                    cost += nearest(s.pendingMedoids, x);
                }
            }

            Partial p = new Partial();
            p.subtask = subtask;
            p.sample = localSample.toArray(new double[0][]);
            p.pendingCost = cost;
            p.state = s;
            out.collect(p);

            stateList.clear();
        }

        private double nearest(double[][] medoids, double[] x) {
            double min = Double.MAX_VALUE;
            for (double[] m : medoids) {
                double d = distance.compute(m, x);
                if (d < min) {
                    min = d;
                }
            }
            return min;
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

    /** Single-task combiner: gathers the round's sample (sorted by subtask), runs local PAM
     *  on it to make the next candidate, scores the PENDING candidate (cost summed across
     *  subtasks), keeps the best, and decides when to stop. */
    private static final class Combine
            implements FlatMapFunction<Partial, Update>, IterationListener<Update> {
        private final int k;
        private final int numSamples;
        private final int sampleSize;
        private final int maxIter;
        private final DistanceMetric distance;
        private transient List<Partial> buffer;
        private transient long lastEpochNanos;

        Combine(int k, int numSamples, int sampleSize, int maxIter, DistanceMetric distance) {
            this.k = k;
            this.numSamples = numSamples;
            this.sampleSize = sampleSize;
            this.maxIter = maxIter;
            this.distance = distance;
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
            long nowNanos = System.nanoTime();
            double roundMs = lastEpochNanos == 0 ? 0.0 : (nowNanos - lastEpochNanos) / 1e6;
            lastEpochNanos = nowNanos;
            buffer.sort(Comparator.comparingInt(p -> p.subtask));
            State s = buffer.get(0).state;

            // 1) score the PENDING candidate (cost summed in subtask order).
            double pendingCost = 0.0;
            for (Partial p : buffer) {
                pendingCost += p.pendingCost;
            }

            // 2) gather this round's sample (subtask order, capped at sampleSize).
            List<double[]> sample = new ArrayList<>();
            for (Partial p : buffer) {
                for (double[] x : p.sample) {
                    if (sample.size() >= sampleSize) {
                        break;
                    }
                    sample.add(x);
                }
            }
            buffer = null;

            State next = s.copy();
            next.round = s.round + 1;

            // Update best with the pending candidate we just scored.
            boolean improved = false;
            if (s.pendingMedoids != null && pendingCost < s.bestCost) {
                next.bestMedoids = s.pendingMedoids;
                next.bestCost = pendingCost;
                improved = true;
            }

            // Draw the next candidate from this round's sample, unless we are done drawing.
            if (s.round < numSamples) {
                double[][] sampleArr = sample.toArray(new double[0][]);
                if (sampleArr.length < k) {
                    throw new IllegalStateException(
                        "CLARA sample too small: got " + sampleArr.length + " points, need k=" + k
                        + " (raise sampleSize or numPoints).");
                }
                KMedoidsModel candidate = new PAM(k, maxIter, distance).fitLocal(sampleArr);
                next.pendingMedoids = candidate.medoids();
            } else {
                next.pendingMedoids = null;
            }

            boolean stop = s.round >= numSamples;
            logger.debug("round={}/{} sampleGathered={} pendingScored={} pendingCost={} bestCost={}{} prevRound={}ms",
                s.round, numSamples, sample.size(),
                s.pendingMedoids != null ? "yes" : "no",
                String.format("%.3f", s.pendingMedoids != null ? pendingCost : Double.NaN),
                String.format("%.3f", next.bestCost == Double.MAX_VALUE ? Double.NaN : next.bestCost),
                improved ? " (NEW BEST)" : "", String.format("%.0f", roundMs));
            if (stop) {
                logger.debug("stop after {} samples, bestCost={}", numSamples, next.bestCost);
            }

            Update u = new Update();
            u.state = next;
            u.stop = stop;
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

    /** Search state on the feedback edge: the round counter, the candidate awaiting cost
     *  evaluation, and the best medoid set found so far. */
    public static final class State implements Serializable {
        public int round;
        public double[][] pendingMedoids;  // candidate from last round's PAM, scored this round
        public double[][] bestMedoids;
        public double bestCost;

        public State() {}

        static State initial() {
            State s = new State();
            s.round = 0;
            s.pendingMedoids = null;
            s.bestMedoids = null;
            s.bestCost = Double.MAX_VALUE;
            return s;
        }

        State copy() {
            State s = new State();
            s.round = round;
            s.pendingMedoids = pendingMedoids;
            s.bestMedoids = bestMedoids;
            s.bestCost = bestCost;
            return s;
        }
    }

    /** Per-subtask round output: this subtask's sample slice + the pending candidate's
     *  partial cost over this subtask's points. */
    public static final class Partial implements Serializable {
        public int subtask;
        public double[][] sample;
        public double pendingCost;
        public State state;

        public Partial() {}
    }

    /** Combined per-round result: the new {@link State} + whether to stop. */
    public static final class Update implements Serializable {
        public State state;
        public boolean stop;

        public Update() {}
    }
}
