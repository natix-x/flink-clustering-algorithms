package clustering.algorithms.kmeans;

import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Clusterer;
import clustering.core.EnvFactory;
import clustering.core.EuclideanGeometry;
import clustering.core.FlinkJobs;
import clustering.core.Geometry;
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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** Bisecting k-means (Steinbach, Karypis &amp; Kumar 2000) — divisive: repeatedly split the
 *  highest-cost leaf with a 2-means run until k leaves exist. Mirror of the Spark
 *  {@code clustering.algorithms.kmeans.BisectingKMeans}, same tree model and same
 *  root-to-leaf labelling (see {@link BisectingKMeansModel}).
 *
 *  <h3>Why this is one job instead of Spark's job-per-split</h3>
 *  The Spark version materialises the point subset of every leaf ({@code persist} per child,
 *  {@code unpersist} of the parent) and runs a fresh distributed k-means per split; its own
 *  source comments flag the cost — an uncached branch point makes Catalyst re-run the 2-means
 *  assignment once per child. Here the ENTIRE search is ONE Flink job: the points are cached
 *  per subtask once ({@link ListStateWithCache}, memory + disk spill) and the CLUSTER TREE
 *  travels the FLIP-176 feedback edge. A point's leaf is recomputed by a root-to-leaf walk
 *  (O(depth) distances) instead of being stored in a per-leaf materialised subset, so no leaf
 *  subset is ever written, re-read or re-derived. Only the target leaf's points contribute to a
 *  split round; the rest are skipped after their (cheap) walk.
 *
 *  <h3>Rounds</h3>
 *  <ol>
 *    <li>{@code INIT_MEAN} — one pass for the global mean: the root centroid.</li>
 *    <li>{@code SEED} — one pass collecting a small deterministic sample of the target leaf,
 *        from which the two split seeds are taken (see {@link Combine#seedFrom}).</li>
 *    <li>{@code LLOYD} — 2-means rounds restricted to the target leaf, until the split
 *        converges or {@code maxIter} rounds are spent.</li>
 *    <li>{@code COMMIT} — one pass with the converged split centroids, giving EXACT child
 *        counts and costs (the split criterion), then the split is written into the tree and
 *        the next target leaf is chosen. Back to {@code SEED}, or stop at k leaves.</li>
 *  </ol>
 *
 *  <h3>Determinism</h3>
 *  Leaves are numbered in left-to-right DFS order, so ids depend only on the tree shape, and
 *  every round's partials are summed in subtask-id order.
 *
 *  One honest limit, worth reporting rather than hiding: Flink's rebalance partitioner starts at
 *  a RANDOM channel, so even at a fixed parallelism the assignment of points to subtasks differs
 *  between runs, and with it the floating-point summation order. That is invisible unless the
 *  split criterion has an exact tie (e.g. congruent, equal-mass leaves), in which case the split
 *  ORDER — and therefore the leaf numbering — may differ between two runs of the same config.
 *  Spark's deterministic partitioning does not have this failure mode. */
public class BisectingKMeans implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(BisectingKMeans.class);

    private static final TypeInformation<State> STATE_TYPE = TypeInformation.of(State.class);
    private static final TypeInformation<Partial> PARTIAL_TYPE = TypeInformation.of(Partial.class);
    private static final TypeInformation<Update> UPDATE_TYPE = TypeInformation.of(Update.class);

    /** Points per subtask offered to the seed picker; the picker itself uses the first
     *  {@link #SEED_SAMPLE_CAP} of the subtask-ordered concatenation. */
    private static final int SEED_SAMPLE_CAP = 256;

    static final int PHASE_INIT_MEAN = 0;
    static final int PHASE_SEED = 1;
    static final int PHASE_LLOYD = 2;
    static final int PHASE_COMMIT = 3;

    private final int k;
    private final int maxIter;
    private final double eps;
    private final long seed;
    private final Geometry geometry;

    public BisectingKMeans(int k, int maxIter, double eps, long seed, Geometry geometry) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1, got " + k);
        }
        this.k = k;
        this.maxIter = maxIter;
        this.eps = eps;
        this.seed = seed;
        this.geometry = geometry;
    }

    public BisectingKMeans(int k, int maxIter, double eps, long seed) {
        this(k, maxIter, eps, seed, EuclideanGeometry.INSTANCE);
    }

    @Override
    public BisectingKMeansModel fit(PointSource source, EnvFactory envs, int parallelism) {
        PointSource prepared = geometry.prepare(source);

        // FLIP-176 iterations add feedback edges (unbounded graph) -> STREAMING mode.
        StreamExecutionEnvironment env = envs.newEnv();
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);

        DataStream<State> initState =
            env.fromCollection(Collections.singletonList(State.initial(k)), STATE_TYPE);
        DataStream<DenseVector> points = prepared.create(env);

        DataStreamList result = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initState),
            ReplayableDataStreamList.notReplay(points),
            IterationConfig.newBuilder().build(),
            new BisectingIterationBody(k, maxIter, eps, geometry));

        State finalState = FlinkJobs.last(result.<State>get(0), "bisectingkmeans-fit");
        if (finalState == null || finalState.nodeCount == 0) {
            throw new RuntimeException("bisecting k-means produced no cluster tree");
        }
        return new BisectingKMeansModel(
            Arrays.copyOf(finalState.centroid, finalState.nodeCount),
            Arrays.copyOf(finalState.left, finalState.nodeCount),
            Arrays.copyOf(finalState.right, finalState.nodeCount),
            Arrays.copyOf(finalState.leafId, finalState.nodeCount),
            geometry.modelDistance());
    }

    /** One round: per-subtask fold (parallel) -> combine (1) -> feed the tree back. */
    private static final class BisectingIterationBody implements IterationBody {
        private final int k;
        private final int maxIter;
        private final double eps;
        private final Geometry geometry;

        BisectingIterationBody(int k, int maxIter, double eps, Geometry geometry) {
            this.k = k;
            this.maxIter = maxIter;
            this.eps = eps;
            this.geometry = geometry;
        }

        @Override
        public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<State> state = variableStreams.get(0);
            DataStream<DenseVector> points = dataStreams.get(0);

            DataStream<Partial> partials = points
                .connect(state.broadcast())
                .transform("bisecting-partial", PARTIAL_TYPE, new SplitFold(geometry.fitDistance()));

            DataStream<Update> updates = partials
                .flatMap(new Combine(k, maxIter, eps, geometry))
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

    /** Parallel per-subtask fold. Caches the local points once, then per round walks each point
     *  down the current tree and contributes only if it lands in the target leaf. */
    private static final class SplitFold
            extends AbstractStreamOperator<Partial>
            implements TwoInputStreamOperator<DenseVector, State, Partial>,
                       IterationListener<Partial> {

        private final DistanceMetric distance;
        private transient ListStateWithCache<double[]> points;
        private transient ListState<State> stateList;

        SplitFold(DistanceMetric distance) {
            this.distance = distance;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            stateList = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("bisecting-state", STATE_TYPE));
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
            Optional<State> current = OperatorStateUtils.getUniqueElement(stateList, "bisecting-state");
            if (!current.isPresent()) {
                return;
            }
            State s = current.get();
            Partial p = new Partial();
            p.subtask = getRuntimeContext().getIndexOfThisSubtask();
            p.state = s;
            p.counts = new long[2];
            p.costs = new double[2];

            if (s.phase == PHASE_INIT_MEAN) {
                foldGlobalMean(p);
            } else if (s.phase == PHASE_SEED) {
                foldSeedSample(s, p);
            } else {
                foldSplit(s, p);
            }

            out.collect(p);
            stateList.clear();
        }

        /** Global coordinate sum + count: the root centroid. */
        private void foldGlobalMean(Partial p) throws Exception {
            for (double[] x : points.get()) {
                if (p.sums == null) {
                    p.sums = new double[2][x.length];
                }
                addInto(p.sums[0], x);
                p.counts[0]++;
            }
        }

        /** The target leaf's row count plus a bounded prefix of its points, for seeding. */
        private void foldSeedSample(State s, Partial p) throws Exception {
            List<double[]> sample = new ArrayList<>();
            for (double[] x : points.get()) {
                if (leafOf(x, s, distance) == s.targetNode) {
                    p.counts[0]++;
                    if (sample.size() < SEED_SAMPLE_CAP) {
                        sample.add(x);
                    }
                }
            }
            p.sample = sample.toArray(new double[0][]);
        }

        /** 2-means partials over the target leaf only: per branch count, coordinate sum and
         *  cost sum d^2 (the split criterion). */
        private void foldSplit(State s, Partial p) throws Exception {
            double[][] split = s.splitCentroids;
            for (double[] x : points.get()) {
                if (leafOf(x, s, distance) != s.targetNode) {
                    continue;
                }
                if (p.sums == null) {
                    p.sums = new double[2][x.length];
                }
                double d0 = distance.compute(x, split[0]);
                double d1 = distance.compute(x, split[1]);
                // Ties go to branch 0, matching the model's left-going tie rule.
                int branch = d0 <= d1 ? 0 : 1;
                double d = branch == 0 ? d0 : d1;
                addInto(p.sums[branch], x);
                p.counts[branch]++;
                p.costs[branch] += d * d;
            }
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

    /** Single-task combiner: sums the round's partials and drives the split search. Its
     *  bookkeeping (per-leaf cost and count, unsplittable leaves, the current split's round
     *  counter) lives in these fields — one instance, one task, all rounds — so only the tree
     *  itself has to travel the feedback edge. */
    private static final class Combine
            implements FlatMapFunction<Partial, Update>, IterationListener<Update> {

        private final int k;
        private final int maxIter;
        private final double eps;
        private final Geometry geometry;

        private transient List<Partial> buffer;
        private transient Map<Integer, Double> leafCost;
        private transient Map<Integer, Long> leafCount;
        private transient Set<Integer> unsplittable;
        private transient int splitRounds;

        Combine(int k, int maxIter, double eps, Geometry geometry) {
            this.k = k;
            this.maxIter = maxIter;
            this.eps = eps;
            this.geometry = geometry;
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
            if (leafCost == null) {
                leafCost = new HashMap<>();
                leafCount = new HashMap<>();
                unsplittable = new HashSet<>();
            }
            buffer.sort(Comparator.comparingInt(p -> p.subtask));
            State s = buffer.get(0).state;

            long[] counts = new long[2];
            double[] costs = new double[2];
            double[][] sums = null;
            List<double[]> sample = new ArrayList<>();
            for (Partial p : buffer) {
                counts[0] += p.counts[0];
                counts[1] += p.counts[1];
                costs[0] += p.costs[0];
                costs[1] += p.costs[1];
                if (p.sums != null) {
                    if (sums == null) {
                        sums = new double[2][p.sums[0].length];
                    }
                    addInto(sums[0], p.sums[0]);
                    addInto(sums[1], p.sums[1]);
                }
                if (p.sample != null) {
                    for (double[] x : p.sample) {
                        if (sample.size() < SEED_SAMPLE_CAP) {
                            sample.add(x);
                        }
                    }
                }
            }
            buffer = null;

            Update u;
            switch (s.phase) {
                case PHASE_INIT_MEAN:
                    u = initRoot(s, counts[0], sums);
                    break;
                case PHASE_SEED:
                    u = startSplit(s, counts[0], sample);
                    break;
                case PHASE_LLOYD:
                    u = lloydRound(s, counts, sums);
                    break;
                default:
                    u = commitSplit(s, counts, costs);
                    break;
            }
            out.collect(u);
        }

        /** Root node = global mean; the only leaf, so it is also the first split target. */
        private Update initRoot(State s, long n, double[][] sums) {
            if (n < k) {
                throw new IllegalStateException(
                    "Dataset too small: n=" + n + " rows but k=" + k + " clusters requested.");
            }
            State next = s.copy();
            int root = next.addNodes(1);
            next.centroid[root] = geometry.project(mean(sums[0], n));
            leafCount.put(0, n);
            // Root cost is never needed: with one leaf the target is forced.
            leafCost.put(0, Double.MAX_VALUE);

            if (k == 1) {
                return stop(next);
            }
            next.phase = PHASE_SEED;
            next.targetNode = 0;
            return cont(next);
        }

        /** Turn the target leaf's sample into the two split seeds. */
        private Update startSplit(State s, long targetRows, List<double[]> sample) {
            State next = s.copy();
            leafCount.put(s.targetNode, targetRows);
            if (sample.size() < 2) {
                // Fewer than two rows, or all rows identical -> nothing to bisect.
                return afterLeafSettled(next, s.targetNode, true);
            }
            double[][] seeds = seedFrom(sample);
            if (seeds == null) {
                return afterLeafSettled(next, s.targetNode, true);
            }
            next.splitCentroids = seeds;
            next.phase = PHASE_LLOYD;
            splitRounds = 0;
            return cont(next);
        }

        /** Two seeds from the leaf's sample: its first point, and the point farthest from it
         *  (farthest-first, a deterministic 2-point k-center).
         *
         *  Deviation from the Spark side, stated on purpose: there the bisection is a full
         *  {@code KMeans} run whose seeds come from a seeded weighted sample of the leaf. Two
         *  well-separated seeds need no RNG, cannot draw the same point twice, and make the
         *  split reproducible without a shuffle. Returns {@code null} when every sampled point
         *  coincides with the first — the leaf is then unsplittable. */
        private double[][] seedFrom(List<double[]> sample) {
            DistanceMetric metric = geometry.fitDistance();
            double[] first = sample.get(0);
            double[] farthest = null;
            double best = 0.0;
            for (double[] x : sample) {
                double d = metric.compute(first, x);
                if (d > best) {
                    best = d;
                    farthest = x;
                }
            }
            return farthest == null ? null : new double[][] {first.clone(), farthest.clone()};
        }

        /** One 2-means round on the target leaf; on convergence the next round is the COMMIT
         *  pass, which re-measures the converged split exactly. */
        private Update lloydRound(State s, long[] counts, double[][] sums) {
            State next = s.copy();
            double[][] split = new double[2][];
            for (int b = 0; b < 2; b++) {
                split[b] = counts[b] == 0L || sums == null
                    ? s.splitCentroids[b]
                    : geometry.project(mean(sums[b], counts[b]));
            }
            double movement = Math.max(
                geometry.fitDistance().compute(s.splitCentroids[0], split[0]),
                geometry.fitDistance().compute(s.splitCentroids[1], split[1]));
            next.splitCentroids = split;
            splitRounds++;
            if (movement < eps || splitRounds >= maxIter) {
                next.phase = PHASE_COMMIT;
            }
            return cont(next);
        }

        /** Write the converged split into the tree (or mark the leaf unsplittable when the
         *  bisection was degenerate), then pick the next target. */
        private Update commitSplit(State s, long[] counts, double[] costs) {
            State next = s.copy();
            if (counts[0] == 0L || counts[1] == 0L) {
                // Degenerate bisection — 2-means put everything in one child. Do not retry.
                logger.warn("bisecting k-means: split of leaf node {} ({} rows) produced an empty "
                    + "child; marking it unsplittable", s.targetNode, leafCount.get(s.targetNode));
                return afterLeafSettled(next, s.targetNode, true);
            }
            int leftNode = next.addNodes(2);
            int rightNode = leftNode + 1;
            next.centroid[leftNode] = s.splitCentroids[0];
            next.centroid[rightNode] = s.splitCentroids[1];
            next.left[s.targetNode] = leftNode;
            next.right[s.targetNode] = rightNode;

            leafCost.remove(s.targetNode);
            leafCount.remove(s.targetNode);
            leafCost.put(leftNode, costs[0]);
            leafCost.put(rightNode, costs[1]);
            leafCount.put(leftNode, counts[0]);
            leafCount.put(rightNode, counts[1]);
            next.splitCentroids = null;
            return afterLeafSettled(next, -1, false);
        }

        /** Common tail: optionally mark a leaf unsplittable, then either target the next
         *  highest-cost splittable leaf or finish. */
        private Update afterLeafSettled(State next, int settledLeaf, boolean markUnsplittable) {
            if (markUnsplittable) {
                unsplittable.add(settledLeaf);
                next.splitCentroids = null;
            }
            if (leafCost.size() >= k) {
                return stop(next);
            }
            // Highest-cost splittable leaf; ties resolve to the lower node index, so the tree
            // is reproducible.
            int target = -1;
            double bestCost = Double.NEGATIVE_INFINITY;
            for (Map.Entry<Integer, Double> e : leafCost.entrySet()) {
                int node = e.getKey();
                double cost = e.getValue();
                boolean splittable = !unsplittable.contains(node)
                    && leafCount.getOrDefault(node, 0L) >= 2L
                    && cost > 0.0;
                if (splittable && (cost > bestCost || (cost == bestCost && node < target))) {
                    bestCost = cost;
                    target = node;
                }
            }
            if (target < 0) {
                logger.warn("bisecting k-means: no splittable leaf left after {} clusters "
                    + "(requested k={}) — stopping early", leafCost.size(), k);
                return stop(next);
            }
            next.phase = PHASE_SEED;
            next.targetNode = target;
            return cont(next);
        }

        /** Freeze the tree: number leaves in left-to-right DFS order. */
        private Update stop(State next) {
            next.leafId = new int[next.centroid.length];
            Arrays.fill(next.leafId, -1);
            numberLeaves(next, 0, new int[] {0});
            Update u = new Update();
            u.state = next;
            u.stop = true;
            return u;
        }

        private static void numberLeaves(State s, int node, int[] nextId) {
            if (s.left[node] < 0) {
                s.leafId[node] = nextId[0]++;
                return;
            }
            numberLeaves(s, s.left[node], nextId);
            numberLeaves(s, s.right[node], nextId);
        }

        private static Update cont(State next) {
            Update u = new Update();
            u.state = next;
            u.stop = false;
            return u;
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

    /** The cluster tree under construction, plus the current split, on the feedback edge.
     *  Flat arrays sized for the final tree (2k-1 nodes); {@code left[i] &lt; 0} marks a leaf. */
    public static final class State implements Serializable {
        public int phase;
        public double[][] centroid;
        public int[] left;
        public int[] right;
        public int nodeCount;
        /** Leaf node currently being bisected, {@code -1} when none. */
        public int targetNode;
        /** The two centroids of the split in progress, {@code null} between splits. */
        public double[][] splitCentroids;
        /** Cluster id per node, filled only in the final state ({@code -1} = internal). */
        public int[] leafId;

        public State() {}

        static State initial(int k) {
            State s = new State();
            s.phase = PHASE_INIT_MEAN;
            // Grown by exactly the nodes that exist: Flink's array serializers are happier
            // without null holes than a pre-sized 2k-1 buffer would be.
            s.centroid = new double[0][];
            s.left = new int[0];
            s.right = new int[0];
            s.nodeCount = 0;
            s.targetNode = -1;
            return s;
        }

        /** Appends {@code extra} fresh LEAF slots and returns the index of the first one. */
        int addNodes(int extra) {
            int first = nodeCount;
            centroid = Arrays.copyOf(centroid, nodeCount + extra);
            left = Arrays.copyOf(left, nodeCount + extra);
            right = Arrays.copyOf(right, nodeCount + extra);
            for (int i = first; i < first + extra; i++) {
                left[i] = -1;
                right[i] = -1;
            }
            nodeCount += extra;
            return first;
        }

        State copy() {
            State s = new State();
            s.phase = phase;
            s.centroid = centroid.clone();
            s.left = left.clone();
            s.right = right.clone();
            s.nodeCount = nodeCount;
            s.targetNode = targetNode;
            s.splitCentroids = splitCentroids;
            s.leafId = leafId;
            return s;
        }
    }

    /** Per-subtask round output. {@code counts}/{@code sums}/{@code costs} are per split
     *  branch, except in the INIT_MEAN round where slot 0 holds the global aggregate;
     *  {@code sample} is only filled in a SEED round. */
    public static final class Partial implements Serializable {
        public int subtask;
        public long[] counts;
        public double[][] sums;
        public double[] costs;
        public double[][] sample;
        public State state;

        public Partial() {}
    }

    /** Combined per-round result: the new {@link State} + whether to stop. */
    public static final class Update implements Serializable {
        public State state;
        public boolean stop;

        public Update() {}
    }

    // ── shared math ───────────────────────────────────────────────────────────

    /** Cluster-tree walk: at each internal node follow the closer child centroid (ties left),
     *  exactly as {@link BisectingKMeansModel#predict} does. */
    static int leafOf(double[] p, State s, DistanceMetric distance) {
        int node = 0;
        while (s.left[node] >= 0) {
            int l = s.left[node];
            int r = s.right[node];
            node = distance.compute(p, s.centroid[l]) <= distance.compute(p, s.centroid[r]) ? l : r;
        }
        return node;
    }

    private static void addInto(double[] acc, double[] p) {
        for (int d = 0; d < acc.length; d++) {
            acc[d] += p[d];
        }
    }

    private static double[] mean(double[] sum, long count) {
        double[] c = new double[sum.length];
        for (int d = 0; d < sum.length; d++) {
            c[d] = sum[d] / count;
        }
        return c;
    }
}
