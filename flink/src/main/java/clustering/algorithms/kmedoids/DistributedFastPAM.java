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
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/** Distributed FastPAM on the Flink ML bounded-iteration framework (FLIP-176) — the whole
 *  training (BUILD + SWAP) is ONE Flink job, mirroring {@link clustering.algorithms.kmeans.FlinkMLKMeans}.
 *  This is the cluster-friendly path: no per-round job submission, points cached once.
 *
 *  <h3>Why FLIP-176 here</h3>
 *  k-medoids needs many rounds (k BUILD steps + up-to-maxIter SWAP rounds). As separate
 *  jobs that is k+T submissions; on FLIP-176 it is one job — the local points are cached
 *  per subtask in a spillable {@link ListStateWithCache} and the medoid state cycles
 *  through a feedback edge.
 *
 *  <h3>What is distributed, and the candidate set</h3>
 *  FastPAM evaluates every non-medoid candidate {@code h} against every point {@code j}.
 *  The points {@code j} are partitioned (cached per subtask); the candidate coordinates
 *  {@code C} (all n points) must be present on EVERY subtask, so they are collected once
 *  and shipped as an operator field in the job graph. Memory is O(n·dim) per subtask — the
 *  O(n²) distance matrix is never built. (Caveat: a very large {@code C} inflates the job
 *  graph; on a real cluster bump {@code akka.framesize} / blob-server limits if needed.)
 *
 *  <h3>BUILD + SWAP as one iteration body</h3>
 *  The fed-back {@link State} carries the medoid indices and {@code numSelected}. While
 *  {@code numSelected < k} the round is a BUILD step (round 0 = first medoid by min total
 *  distance; later = max cost reduction). Once {@code numSelected == k} every round is a
 *  FastPAM1 SWAP round. Each subtask folds its local points into per-candidate partials;
 *  the parallelism-1 combiner sums them (sorted by subtask id for determinism), picks the
 *  move, and feeds back the new {@link State}; iteration stops at a SWAP local optimum or
 *  after {@code maxIter} swap rounds.
 *
 *  <h3>Determinism</h3>
 *  Reproducible for a FIXED parallelism (partials summed in subtask-id order). Across
 *  different parallelism the j-partition changes, so floating-point sum order differs and
 *  near-tie moves may resolve differently — unavoidable for a distributed sum. */
public class DistributedFastPAM implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(DistributedFastPAM.class);

    private static final TypeInformation<State> STATE_TYPE = TypeInformation.of(State.class);
    private static final TypeInformation<Partial> PARTIAL_TYPE = TypeInformation.of(Partial.class);
    private static final TypeInformation<Update> UPDATE_TYPE = TypeInformation.of(Update.class);

    private final int k;
    private final int maxIter;
    private final DistanceMetric distance;

    public DistributedFastPAM(int k, int maxIter, DistanceMetric distance) {
        this.k = k;
        this.maxIter = maxIter;
        this.distance = distance;
    }

    @Override
    public Model fit(PointSource source, EnvFactory envs, int parallelism) {
        // Candidate coordinates: collected once (deterministic index order), then shipped
        // with the operators in the single iteration job. O(n·dim) — NOT the O(n²) matrix.
        double[][] c = Datasets.collectAll(source, envs).toArray(new double[0][]);
        int n = c.length;
        if (n < k) {
            throw new IllegalArgumentException(
                "Dataset too small: n=" + n + " points but k=" + k + " medoids requested.");
        }

        // FLIP-176 iterations add feedback edges (unbounded graph) -> STREAMING mode.
        StreamExecutionEnvironment env = envs.newEnv();
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);

        DataStream<State> initState =
            env.fromCollection(java.util.Collections.singletonList(State.initial(k)), STATE_TYPE);
        DataStream<double[]> points = source.create(env);

        DataStreamList result = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initState),
            ReplayableDataStreamList.notReplay(points),
            IterationConfig.newBuilder().build(),
            new SwapIterationBody(c, k, maxIter, distance));

        State finalState = FlinkJobs.last(result.<State>get(0), "distfastpam-fit");

        int[] medoids = (finalState == null) ? firstK(n, k) : finalState.medoids;
        double[][] medoidCoords = new double[k][];
        for (int i = 0; i < k; i++) {
            medoidCoords[i] = c[medoids[i]];
        }
        return new KMedoidsModel(medoidCoords, distance);
    }

    /** One round: per-subtask fold (parallel) -> combine (1) -> feed back medoid state;
     *  stop at SWAP local optimum or after maxIter swap rounds. */
    private static final class SwapIterationBody implements IterationBody {
        private final double[][] c;
        private final int k;
        private final int maxIter;
        private final DistanceMetric distance;

        SwapIterationBody(double[][] c, int k, int maxIter, DistanceMetric distance) {
            this.c = c;
            this.k = k;
            this.maxIter = maxIter;
            this.distance = distance;
        }

        @Override
        public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<State> state = variableStreams.get(0);
            DataStream<double[]> points = dataStreams.get(0);

            DataStream<Partial> partials = points
                .connect(state.broadcast())
                .transform("dfastpam-partial", PARTIAL_TYPE, new PartialFold(c, k, distance));

            DataStream<Update> updates = partials
                .flatMap(new Combine(c.length, k, maxIter))
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

    /** Parallel per-subtask fold. Local points cached in a spillable {@link ListStateWithCache};
     *  the candidate coords {@code C} are an operator field (shipped once). Each epoch it reads
     *  the broadcast {@link State} and emits its per-candidate partial array — BUILD shape
     *  {@code [n]} or SWAP shape {@code [n + n*k]} depending on {@code numSelected}. */
    private static final class PartialFold
            extends AbstractStreamOperator<Partial>
            implements TwoInputStreamOperator<double[], State, Partial>,
                       IterationListener<Partial> {

        private final double[][] c;
        private final int k;
        private final int n;
        private final DistanceMetric distance;
        private transient ListStateWithCache<double[]> points;
        private transient ListState<State> stateList;

        PartialFold(double[][] c, int k, DistanceMetric distance) {
            this.c = c;
            this.k = k;
            this.n = c.length;
            this.distance = distance;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            stateList = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("dfastpam-state", STATE_TYPE));
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
            Optional<State> current = OperatorStateUtils.getUniqueElement(stateList, "dfastpam-state");
            if (!current.isPresent()) {
                return;
            }
            State s = current.get();
            int subtask = getRuntimeContext().getIndexOfThisSubtask();

            double[] acc;
            if (s.numSelected < k) {
                acc = foldBuild(s);
            } else {
                acc = foldSwap(s);
            }

            Partial p = new Partial();
            p.subtask = subtask;
            p.data = acc;
            p.state = s;
            out.collect(p);

            stateList.clear();
        }

        /** BUILD fold: total distance (first medoid) or cost-reduction gain. Length n. */
        private double[] foldBuild(State s) throws Exception {
            double[] acc = new double[n];
            if (s.numSelected == 0) {
                for (double[] x : points.get()) {
                    for (int h = 0; h < n; h++) {
                        acc[h] += distance.compute(c[h], x);
                    }
                }
            } else {
                double[][] selected = gather(c, s.medoids, s.numSelected);
                for (double[] x : points.get()) {
                    double dBest = Double.MAX_VALUE;
                    for (double[] sel : selected) {
                        double d = distance.compute(sel, x);
                        if (d < dBest) {
                            dBest = d;
                        }
                    }
                    for (int h = 0; h < n; h++) {
                        double g = dBest - distance.compute(c[h], x);
                        if (g > 0.0) {
                            acc[h] += g;
                        }
                    }
                }
            }
            return acc;
        }

        /** SWAP fold: FastPAM1 shared[n] + removeLoss[n*k]. Length n + n*k. */
        private double[] foldSwap(State s) throws Exception {
            double[][] medoids = gather(c, s.medoids, k);
            double[] acc = new double[n + n * k];
            for (double[] x : points.get()) {
                double d1 = Double.MAX_VALUE;
                double d2 = Double.MAX_VALUE;
                int n1 = -1;
                for (int m = 0; m < k; m++) {
                    double d = distance.compute(medoids[m], x);
                    if (d < d1) {
                        d2 = d1;
                        d1 = d;
                        n1 = m;
                    } else if (d < d2) {
                        d2 = d;
                    }
                }
                for (int h = 0; h < n; h++) {
                    double dj = distance.compute(c[h], x);
                    double sharedContrib = dj < d1 ? dj - d1 : 0.0;
                    acc[h] += sharedContrib;
                    double caseB = Math.min(d2, dj) - d1;
                    acc[n + h * k + n1] += (caseB - sharedContrib);
                }
            }
            return acc;
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

    /** Single-task combiner: sums per-subtask partials (sorted by subtask id), then picks
     *  the BUILD selection or the best SWAP, and decides whether to stop. */
    private static final class Combine
            implements FlatMapFunction<Partial, Update>, IterationListener<Update> {
        private final int n;
        private final int k;
        private final int maxIter;
        private transient List<Partial> buffer;
        private transient long lastEpochNanos;

        Combine(int n, int k, int maxIter) {
            this.n = n;
            this.k = k;
            this.maxIter = maxIter;
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
            String phase = s.numSelected < k ? "BUILD" : "SWAP";
            logger.debug("epoch={} phase={} numSelected={}/{} swapRounds={}/{} partials={} prevRound={}ms",
                epoch, phase, s.numSelected, k, s.swapRounds, maxIter, buffer.size(), String.format("%.0f", roundMs));

            int len = buffer.get(0).data.length;
            double[] sum = new double[len];
            for (Partial p : buffer) {
                for (int i = 0; i < len; i++) {
                    sum[i] += p.data[i];
                }
            }
            buffer = null;

            State next = s.copy();
            boolean stop;

            if (s.numSelected < k) {
                // BUILD: choose the next medoid.
                boolean[] excluded = new boolean[n];
                for (int i = 0; i < s.numSelected; i++) {
                    excluded[s.medoids[i]] = true;
                }
                int idx = (s.numSelected == 0)
                    ? argMinExcluding(sum, excluded)
                    : argMaxExcluding(sum, excluded);
                next.medoids[s.numSelected] = idx;
                next.numSelected = s.numSelected + 1;
                stop = false; // keep going: more BUILD steps, then SWAP
                logger.debug("  BUILD picked medoid #{} = point[{}]  (now {}/{} selected)",
                    s.numSelected, idx, next.numSelected, k);
            } else {
                // SWAP: best FastPAM1 move (ascending h, then ascending i, strict improvement).
                boolean[] isMedoid = new boolean[n];
                for (int i = 0; i < k; i++) {
                    isMedoid[s.medoids[i]] = true;
                }
                double bestDelta = 0.0;
                int bestH = -1;
                int bestI = -1;
                for (int h = 0; h < n; h++) {
                    if (isMedoid[h]) {
                        continue;
                    }
                    double shared = sum[h];
                    int base = n + h * k;
                    for (int i = 0; i < k; i++) {
                        double delta = shared + sum[base + i];
                        if (delta < bestDelta) {
                            bestDelta = delta;
                            bestH = h;
                            bestI = i;
                        }
                    }
                }
                next.swapRounds = s.swapRounds + 1;
                boolean applied = bestDelta < 0.0;
                if (applied) {
                    next.medoids[bestI] = bestH;
                }
                stop = !applied || next.swapRounds >= maxIter;
                logger.debug("  SWAP round {}/{}{}  bestDelta={}", next.swapRounds, maxIter,
                    applied ? "  swap medoid[" + bestI + "] <- point[" + bestH + "]" : "  no improving swap",
                    bestDelta);
                if (stop) {
                    logger.debug("stop at swapRound={} (bestDelta={}, maxIter={})", next.swapRounds, bestDelta, maxIter);
                }
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

    /** Medoid search state on the feedback edge. While {@code numSelected < k} the next
     *  round is a BUILD step; once equal to k, rounds are SWAP rounds. */
    public static final class State implements Serializable {
        public int[] medoids;     // k slots; entries [0, numSelected) are valid
        public int numSelected;
        public int swapRounds;

        public State() {}

        static State initial(int k) {
            State s = new State();
            s.medoids = new int[k];
            Arrays.fill(s.medoids, -1);
            s.numSelected = 0;
            s.swapRounds = 0;
            return s;
        }

        State copy() {
            State s = new State();
            s.medoids = medoids.clone();
            s.numSelected = numSelected;
            s.swapRounds = swapRounds;
            return s;
        }
    }

    /** Per-subtask partial array for one round, tagged with subtask id (deterministic sum)
     *  and the {@link State} it was computed against. */
    public static final class Partial implements Serializable {
        public int subtask;
        public double[] data;
        public State state;

        public Partial() {}
    }

    /** Combined per-round result: the new medoid {@link State} + whether to stop. */
    public static final class Update implements Serializable {
        public State state;
        public boolean stop;

        public Update() {}
    }

    // ── small helpers ─────────────────────────────────────────────────────────

    private static double[][] gather(double[][] c, int[] indices, int count) {
        double[][] out = new double[count][];
        for (int i = 0; i < count; i++) {
            out[i] = c[indices[i]];
        }
        return out;
    }

    private static int[] firstK(int n, int k) {
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            out[i] = i;
        }
        return out;
    }

    private static int argMinExcluding(double[] v, boolean[] excluded) {
        int best = -1;
        double min = Double.MAX_VALUE;
        for (int i = 0; i < v.length; i++) {
            if (!excluded[i] && v[i] < min) {
                min = v[i];
                best = i;
            }
        }
        return best;
    }

    private static int argMaxExcluding(double[] v, boolean[] excluded) {
        int best = -1;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < v.length; i++) {
            if (!excluded[i] && v[i] > max) {
                max = v[i];
                best = i;
            }
        }
        return best;
    }
}