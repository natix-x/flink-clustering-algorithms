package clustering.algorithms.kmeans;

import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.Geometry;
import clustering.core.PointSource;
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
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
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
import java.util.Random;

/** The shared centroid-iteration kernel: ONE Flink job in which a prototype set cycles
 *  through a FLIP-176 feedback edge and every round is one distributed pass over the points.
 *
 *  Counterpart of the Spark repo's {@code LloydKMeans} object — same role (the loop that
 *  {@code KMeans}, {@code BreathingKMeans} and {@code BisectingKMeans} all share), but the
 *  Flink version is deliberately MORE general in one way and MORE restricted in another:
 *
 *  <ul>
 *    <li>More general: the per-round decision is a pluggable {@link RoundDriver} that lives in
 *        the parallelism-1 combiner. Plain Lloyd, Fritzke's breathing cycle and the bisecting
 *        split search are all drivers over the SAME job, so all three run as a single Flink
 *        job instead of Spark's "one job per iteration, one Lloyd call per phase". The driver
 *        may change the number of centroids between rounds (breathing needs k+m), so nothing
 *        here is fixed to k.</li>
 *    <li>More restricted: the driver runs on ONE task, so its per-round work must stay
 *        O(k·d) — exactly what a Spark driver does between jobs.</li>
 *  </ul>
 *
 *  Each round the parallel {@code PartialAssign} operator folds its cached local points into
 *  {@link RoundStats}: per centroid the point count, the coordinate sum, {@code sum d1^2} and
 *  {@code sum (d2^2 - d1^2)}. Counts+sums are what Lloyd needs; the other two are Fritzke's
 *  error and utility, computed in the same scan because the second-nearest centroid costs
 *  nothing extra once the nearest is known. Local points live in a {@link ListStateWithCache}
 *  (memory + disk spill) so datasets larger than worker heap do not OOM.
 *
 *  <h3>Weights</h3>
 *  The Flink seam carries {@link DenseVector} points with no weight channel (the Spark side
 *  has an optional {@code weight} column), so all statistics here are unweighted — identical
 *  to a Spark run whose input has no weight column.
 *
 *  <h3>Determinism</h3>
 *  Reproducible for a FIXED parallelism: partials are summed in subtask-id order. Across
 *  different parallelism the point partition changes, so floating-point sum order differs and
 *  near-ties may resolve differently — unavoidable for a distributed sum. */
public final class CentroidIteration {

    /** Prototype sets travel as {@code DenseVector[]} — same shape Flink ML's own iteration
     *  uses for centroids, and the counterpart of Spark's {@code Array[Vector]}. */
    private static final TypeInformation<DenseVector[]> CENTROIDS_TYPE =
        ObjectArrayTypeInfo.getInfoFor(DenseVectorTypeInfo.INSTANCE);
    private static final TypeInformation<Partial> PARTIAL_TYPE = TypeInformation.of(Partial.class);
    private static final TypeInformation<Update> UPDATE_TYPE = TypeInformation.of(Update.class);

    private CentroidIteration() {}

    /** Decides, from one round's statistics, which prototype set to evaluate next and whether
     *  this was the last round. Runs in the parallelism-1 combiner and may keep mutable state
     *  in its own fields across rounds (one instance, one task, all rounds).
     *
     *  On {@code stop} the returned centroids are the RESULT of the whole job — which is why
     *  breathing k-means can hand back its best-so-far set rather than the last one tried. */
    public interface RoundDriver extends Serializable {
        Decision nextRound(int epoch, RoundStats stats);
    }

    /** A driver's verdict for one round. */
    public static final class Decision {
        public final DenseVector[] centroids;
        public final boolean stop;

        public Decision(DenseVector[] centroids, boolean stop) {
            this.centroids = centroids;
            this.stop = stop;
        }

        public static Decision cont(DenseVector[] centroids) {
            return new Decision(centroids, false);
        }

        public static Decision stop(DenseVector[] centroids) {
            return new Decision(centroids, true);
        }
    }

    /** Everything one distributed pass yields about the prototype set it was run against.
     *  Indexed by centroid; {@code centroids} is the set the numbers belong to.
     *
     *  - {@code counts[i]}    — points in centroid i's Voronoi cell
     *  - {@code sums[i]}      — coordinate sum of those points (Lloyd's numerator)
     *  - {@code errors[i]}    — phi(c_i) = sum d1^2, Fritzke's breathe-in criterion
     *  - {@code utilities[i]} — U(c_i) = sum (d2^2 - d1^2), the error increase deleting c_i
     *                           would cause, i.e. the breathe-out criterion */
    public static final class RoundStats {
        public final DenseVector[] centroids;
        public final long[] counts;
        public final double[][] sums;
        public final double[] errors;
        public final double[] utilities;

        RoundStats(DenseVector[] centroids, long[] counts, double[][] sums, double[] errors, double[] utilities) {
            this.centroids = centroids;
            this.counts = counts;
            this.sums = sums;
            this.errors = errors;
            this.utilities = utilities;
        }

        /** phi(C, X) = sum over points of d(x, nearest centroid)^2 — the k-means objective. */
        public double totalError() {
            double total = 0.0;
            for (double e : errors) {
                total += e;
            }
            return total;
        }

        public long totalCount() {
            long total = 0L;
            for (long c : counts) {
                total += c;
            }
            return total;
        }

        /** Lloyd update: the mean of each cell, projected onto {@code geometry}. An empty cell
         *  keeps its old centroid (same fallback as the Spark implementation). */
        public DenseVector[] means(Geometry geometry) {
            DenseVector[] next = new DenseVector[centroids.length];
            for (int i = 0; i < centroids.length; i++) {
                if (counts[i] == 0L) {
                    next[i] = centroids[i];
                } else {
                    double[] mean = new double[sums[i].length];
                    for (int d = 0; d < mean.length; d++) {
                        mean[d] = sums[i][d] / counts[i];
                    }
                    next[i] = geometry.project(new DenseVector(mean));
                }
            }
            return next;
        }
    }

    /** Runs the iteration to completion and returns the driver's final prototype set. */
    public static DenseVector[] run(
            PointSource preparedSource,
            EnvFactory envs,
            DenseVector[] initialCentroids,
            DistanceMetric fitDistance,
            RoundDriver driver,
            String jobName) {
        if (initialCentroids.length == 0) {
            throw new IllegalArgumentException("CentroidIteration: initial centroids must not be empty");
        }
        // FLIP-176 iterations add feedback edges (unbounded graph) -> STREAMING mode.
        StreamExecutionEnvironment env = envs.newEnv();
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);

        DataStream<DenseVector[]> initStream =
            env.fromCollection(Collections.singletonList(initialCentroids), CENTROIDS_TYPE);
        DataStream<DenseVector> points = preparedSource.create(env);

        DataStreamList result = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initStream),
            ReplayableDataStreamList.notReplay(points),
            IterationConfig.newBuilder().build(),
            new CentroidIterationBody(fitDistance, driver));

        DenseVector[] finalCentroids = FlinkJobs.last(result.<DenseVector[]>get(0), jobName);
        return finalCentroids == null ? initialCentroids : finalCentroids;
    }

    /** One round: assign (parallel) -> drive (parallelism 1) -> feed back. */
    private static final class CentroidIterationBody implements IterationBody {
        private final DistanceMetric distance;
        private final RoundDriver driver;

        CentroidIterationBody(DistanceMetric distance, RoundDriver driver) {
            this.distance = distance;
            this.driver = driver;
        }

        @Override
        public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<DenseVector[]> centroids = variableStreams.get(0);
            DataStream<DenseVector> points = dataStreams.get(0);

            DataStream<Partial> partials = points
                .connect(centroids.broadcast())
                .transform("centroid-partial", PARTIAL_TYPE, new PartialAssign(distance));

            DataStream<Update> updates = partials
                .flatMap(new Combine(driver))
                .setParallelism(1)
                .returns(UPDATE_TYPE);

            // Feedback parallelism must match the initial variable stream (1).
            DataStream<DenseVector[]> newCentroids = updates
                .map((MapFunction<Update, DenseVector[]>) u -> u.centroids)
                .returns(CENTROIDS_TYPE)
                .setParallelism(1);

            DataStream<Integer> termination = updates
                .flatMap(new ContinueUnlessStop())
                .returns(Types.INT)
                .setParallelism(1);

            return new IterationBodyResult(
                DataStreamList.of(newCentroids),
                DataStreamList.of(newCentroids),
                termination);
        }
    }

    /** Parallel per-subtask assignment. Local points are cached in a spillable
     *  {@link ListStateWithCache} (memory + disk) so huge datasets do not OOM; the current
     *  centroids arrive on the broadcast side and are kept in operator list state. Low-level
     *  operator because {@link ListStateWithCache} needs the operator's task/state context.
     *
     *  Nothing here is bound to k: the emitted arrays are sized from the centroid set of the
     *  round, so a driver may grow or shrink the set between rounds. */
    private static final class PartialAssign
            extends AbstractStreamOperator<Partial>
            implements TwoInputStreamOperator<DenseVector, DenseVector[], Partial>,
                       IterationListener<Partial> {

        private final DistanceMetric distance;
        private transient ListStateWithCache<DenseVector> points;
        private transient ListState<DenseVector[]> centroids;

        PartialAssign(DistanceMetric distance) {
            this.distance = distance;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            centroids = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("centroids", CENTROIDS_TYPE));
            points = new ListStateWithCache<>(
                DenseVectorTypeInfo.INSTANCE.createSerializer(getExecutionConfig()),
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
            points.add(record.getValue());
        }

        @Override
        public void processElement2(StreamRecord<DenseVector[]> record) throws Exception {
            centroids.add(record.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<Partial> out) throws Exception {
            Optional<DenseVector[]> current = OperatorStateUtils.getUniqueElement(centroids, "centroids");
            if (!current.isPresent()) {
                return;
            }
            DenseVector[] c = current.get();
            int k = c.length;
            int dim = c[0].size();

            Partial partial = new Partial();
            partial.subtask = getRuntimeContext().getIndexOfThisSubtask();
            partial.centroids = c;
            partial.counts = new long[k];
            partial.sums = new double[k][dim];
            partial.errors = new double[k];
            partial.utilities = new double[k];

            for (DenseVector p : points.get()) {
                // Nearest and second nearest in one scan: d2 feeds Fritzke's utility.
                int nearest = 0;
                double d1 = Double.MAX_VALUE;
                double d2 = Double.MAX_VALUE;
                for (int i = 0; i < k; i++) {
                    double d = distance.compute(p, c[i]);
                    if (d < d1) {
                        d2 = d1;
                        d1 = d;
                        nearest = i;
                    } else if (d < d2) {
                        d2 = d;
                    }
                }
                // A single centroid has no second nearest: utility 0, not +inf.
                double d2Squared = (d2 == Double.MAX_VALUE) ? d1 * d1 : d2 * d2;
                double[] sum = partial.sums[nearest];
                double[] coords = p.values;
                for (int d = 0; d < dim; d++) {
                    sum[d] += coords[d];
                }
                partial.counts[nearest]++;
                partial.errors[nearest] += d1 * d1;
                partial.utilities[nearest] += d2Squared - d1 * d1;
            }
            out.collect(partial);
            centroids.clear();
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Partial> out) throws Exception {
            points.clear();
        }

        @Override public void processWatermark1(Watermark mark) {}
        @Override public void processWatermark2(Watermark mark) {}
        @Override public void processLatencyMarker1(LatencyMarker latencyMarker) {}
        @Override public void processLatencyMarker2(LatencyMarker latencyMarker) {}
        // processWatermarkStatus1/2 are final in AbstractStreamOperator — not overridden.
    }

    /** Single-task combiner: merges the round's partials (in subtask-id order, so the sum is
     *  deterministic) into {@link RoundStats} and hands them to the {@link RoundDriver}.
     *  Buffers only k-sized partials, so no OOM risk. */
    private static final class Combine
            implements FlatMapFunction<Partial, Update>, IterationListener<Update> {

        private final RoundDriver driver;
        private transient List<Partial> buffer;

        Combine(RoundDriver driver) {
            this.driver = driver;
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
            buffer.sort(Comparator.comparingInt(p -> p.subtask));
            DenseVector[] centroids = buffer.get(0).centroids;
            int k = centroids.length;
            int dim = centroids[0].size();

            long[] counts = new long[k];
            double[][] sums = new double[k][dim];
            double[] errors = new double[k];
            double[] utilities = new double[k];
            for (Partial p : buffer) {
                for (int i = 0; i < k; i++) {
                    counts[i] += p.counts[i];
                    errors[i] += p.errors[i];
                    utilities[i] += p.utilities[i];
                    for (int d = 0; d < dim; d++) {
                        sums[i][d] += p.sums[i][d];
                    }
                }
            }
            buffer = null;

            Decision decision = driver.nextRound(epoch, new RoundStats(centroids, counts, sums, errors, utilities));
            Update update = new Update();
            update.centroids = decision.centroids;
            update.stop = decision.stop;
            out.collect(update);
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Update> out) {}
    }

    /** Emits a "continue" token each round unless the round's {@link Update} says stop.
     *  An empty round here tells the iteration framework to terminate. */
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

    /** Per-subtask partial aggregate for one round. POJO for Flink serialization. */
    public static final class Partial implements Serializable {
        public int subtask;
        public DenseVector[] centroids;
        public long[] counts;
        public double[][] sums;
        public double[] errors;
        public double[] utilities;

        public Partial() {}
    }

    /** Combined per-round result: the prototype set for the next round + whether to stop. */
    public static final class Update implements Serializable {
        public DenseVector[] centroids;
        public boolean stop;

        public Update() {}
    }

    // ── helpers shared by the drivers ─────────────────────────────────────────

    /** Seeded sample of {@code k} distinct starting centroids, taken from ALREADY PREPARED
     *  data. Kept here so every centroid algorithm initialises identically.
     *
     *  Deviation from the Spark side, on purpose: Spark draws the sample with a weighted
     *  reservoir over the whole DataFrame ({@code rand(seed)^(1/w)} + {@code orderBy}), which
     *  costs a full shuffle. Here the candidates are the deterministic head of the source
     *  (parallelism 1, source order — see {@link Datasets#collectHead}) and the k picks are
     *  drawn from that head with a seeded RNG. Both are "a seeded sample of the data"; this
     *  one is reproducible across parallelism settings, which the reproducibility measurements
     *  need, and avoids a shuffle whose only purpose is picking k rows. */
    public static DenseVector[] sampleInitialCentroids(PointSource preparedSource, EnvFactory envs, int k, long seed) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1, got " + k);
        }
        List<DenseVector> candidates = Datasets.collectHead(preparedSource, envs, Math.max(k * 30L, 1000L));
        int n = candidates.size();
        if (n < k) {
            throw new IllegalArgumentException(
                "Could not sample " + k + " initial centroids — too few points (n=" + n + ").");
        }
        Random rng = new Random(seed);
        DenseVector[] centroids = new DenseVector[k];
        boolean[] taken = new boolean[n];
        int picked = 0;
        while (picked < k) {
            int idx = rng.nextInt(n);
            if (!taken[idx]) {
                taken[idx] = true;
                centroids[picked++] = new DenseVector(candidates.get(idx).values.clone());
            }
        }
        return centroids;
    }

    /** Largest centroid movement between two equally-sized sets, under {@code distance}. */
    public static double maxMovement(DenseVector[] from, DenseVector[] to, DistanceMetric distance) {
        double max = 0.0;
        for (int i = 0; i < from.length; i++) {
            max = Math.max(max, distance.compute(from[i], to[i]));
        }
        return max;
    }
}
