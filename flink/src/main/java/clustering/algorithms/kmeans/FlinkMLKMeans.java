package clustering.algorithms.kmeans;

import clustering.core.Clusterer;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.Model;
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

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Random;

/** KMeans on the Flink ML bounded-iteration framework (FLIP-176). The algorithm
 *  is ours (init / assignment / mean update / empty-cluster handling / convergence);
 *  only the iteration runtime is Flink ML's. The whole training is ONE Flink job:
 *  centroids cycle through a feedback edge.
 *
 *  Distance work is DISTRIBUTED. Each round:
 *    - {@code PartialAssign} (parallel): every subtask assigns its local points to
 *      the broadcast centroids and emits per-cluster partial sums + counts. Local
 *      points are cached in a {@link ListStateWithCache} (memory + disk spill), so
 *      datasets larger than worker heap don't OOM — same approach as flink-ml KMeans.
 *    - {@code CombinePartials} (parallelism 1, cheap k-sized merge): builds the new
 *      centroids (empty clusters keep their old position), checks convergence
 *      (max centroid movement < eps) and whether maxIter is reached, emits an
 *      {@code Update}.
 *  The {@code Update} feeds two streams: the centroids feedback, and the termination
 *  criteria (a "continue" token unless converged / maxIter reached). */
public class FlinkMLKMeans implements Clusterer {

    private static final TypeInformation<double[][]> CENTROIDS_TYPE =
        Types.OBJECT_ARRAY(PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO);
    private static final TypeInformation<Partial> PARTIAL_TYPE = TypeInformation.of(Partial.class);
    private static final TypeInformation<Update> UPDATE_TYPE = TypeInformation.of(Update.class);

    private final int k;
    private final int maxIter;
    private final double eps;
    private final DistanceMetric distance;
    private final long seed;

    public FlinkMLKMeans(int k, int maxIter, double eps, DistanceMetric distance, long seed) {
        this.k = k;
        this.maxIter = maxIter;
        this.eps = eps;
        this.distance = distance;
        this.seed = seed;
    }

    @Override
    public Model fit(PointSource source, EnvFactory envs, int parallelism) {
        // Initial centroids from a deterministic head sample (index order, parallelism 1).
        List<double[]> sample = Datasets.collectHead(source, envs, Math.max(k * 30L, 1000L));
        double[][] init = pick(sample, k, seed);

        // Flink ML iterations add feedback edges (unbounded graph) -> STREAMING mode.
        StreamExecutionEnvironment env = envs.newEnv();
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);
        DataStream<double[][]> initCentroids =
            env.fromCollection(Collections.singletonList(init), CENTROIDS_TYPE);
        DataStream<double[]> points = source.create(env);

        DataStreamList result = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initCentroids),
            ReplayableDataStreamList.notReplay(points),
            IterationConfig.newBuilder().build(),
            new KMeansIterationBody(k, maxIter, eps, distance));

        double[][] centroids = FlinkJobs.last(result.<double[][]>get(0), "kmeans-fit");
        return new KMeansModel(centroids == null ? init : centroids, distance);
    }

    /** One round: assign (parallel) -> combine (1) -> feed back; stop on convergence
     *  or after maxIter. */
    private static final class KMeansIterationBody implements IterationBody {
        private final int k;
        private final int maxIter;
        private final double eps;
        private final DistanceMetric distance;

        KMeansIterationBody(int k, int maxIter, double eps, DistanceMetric distance) {
            this.k = k;
            this.maxIter = maxIter;
            this.eps = eps;
            this.distance = distance;
        }

        @Override
        public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<double[][]> centroids = variableStreams.get(0);
            DataStream<double[]> points = dataStreams.get(0);

            DataStream<Partial> partials = points
                .connect(centroids.broadcast())
                .transform("kmeans-partial", PARTIAL_TYPE, new PartialAssign(k, distance));

            DataStream<Update> updates = partials
                .flatMap(new CombinePartials(k, maxIter, eps, distance))
                .setParallelism(1)
                .returns(UPDATE_TYPE);

            // Feedback parallelism must match the initial variable stream (1).
            DataStream<double[][]> newCentroids = updates
                .map((MapFunction<Update, double[][]>) u -> u.centroids)
                .returns(CENTROIDS_TYPE)
                .setParallelism(1);

            // terminationCriteria: a "continue" token each round unless we should stop.
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
     *  {@link ListStateWithCache} (memory + disk) so huge datasets don't OOM; the
     *  current centroids are kept in operator list state. At each epoch (round) the
     *  subtask emits its per-cluster partial sums + counts. Low-level operator
     *  because {@link ListStateWithCache} needs the operator's task/state context. */
    private static final class PartialAssign
            extends AbstractStreamOperator<Partial>
            implements TwoInputStreamOperator<double[], double[][], Partial>,
                       IterationListener<Partial> {

        private final int k;
        private final DistanceMetric distance;
        private transient ListStateWithCache<double[]> points;
        private transient ListState<double[][]> centroids;

        PartialAssign(int k, DistanceMetric distance) {
            this.k = k;
            this.distance = distance;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            centroids = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("centroids", CENTROIDS_TYPE));
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
        public void processElement2(StreamRecord<double[][]> record) throws Exception {
            centroids.add(record.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<Partial> out) throws Exception {
            Optional<double[][]> current = OperatorStateUtils.getUniqueElement(centroids, "centroids");
            if (!current.isPresent()) {
                return;
            }
            double[][] c = current.get();
            int dim = c[0].length;
            Partial partial = new Partial();
            partial.counts = new long[k];
            partial.sums = new double[k][dim];
            partial.centroids = c;
            for (double[] p : points.get()) {
                int idx = nearest(p, c, distance);
                addInto(partial.sums[idx], p);
                partial.counts[idx]++;
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

    /** Single-task combiner: merges all subtasks' partials into the new centroids
     *  and decides whether to stop (convergence or maxIter). Buffers only k-sized
     *  partials, so no OOM risk. */
    private static final class CombinePartials
            implements FlatMapFunction<Partial, Update>, IterationListener<Update> {
        private final int k;
        private final int maxIter;
        private final double eps;
        private final DistanceMetric distance;
        private transient long[] counts;
        private transient double[][] sums;
        private transient double[][] centroids;

        CombinePartials(int k, int maxIter, double eps, DistanceMetric distance) {
            this.k = k;
            this.maxIter = maxIter;
            this.eps = eps;
            this.distance = distance;
        }

        @Override
        public void flatMap(Partial partial, Collector<Update> out) {
            if (counts == null) {
                counts = new long[k];
                sums = new double[k][partial.centroids[0].length];
                centroids = partial.centroids;
            }
            for (int i = 0; i < k; i++) {
                counts[i] += partial.counts[i];
                addInto(sums[i], partial.sums[i]);
            }
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<Update> out) {
            if (counts == null) {
                return;
            }
            double[][] next = new double[k][];
            double maxMove = 0.0;
            for (int i = 0; i < k; i++) {
                next[i] = counts[i] == 0 ? centroids[i] : mean(sums[i], counts[i]);
                maxMove = Math.max(maxMove, distance.compute(centroids[i], next[i]));
            }
            Update update = new Update();
            update.centroids = next;
            update.stop = maxMove < eps || epoch >= maxIter - 1;
            if (update.stop) {
                System.out.println("[FlinkMLKMeans] stop at round=" + epoch
                    + " (maxMove=" + maxMove + ", eps=" + eps + ", maxIter=" + maxIter + ")");
            }
            out.collect(update);
            counts = null;
            sums = null;
            centroids = null;
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
        public long[] counts;
        public double[][] sums;
        public double[][] centroids;

        public Partial() {}
    }

    /** Combined per-round result: the new centroids + whether to stop iterating. */
    public static final class Update implements Serializable {
        public double[][] centroids;
        public boolean stop;

        public Update() {}
    }

    // --- shared math ---------------------------------------------------------

    private static double[][] pick(List<double[]> candidates, int k, long seed) {
        int n = candidates.size();
        if (n < k) {
            throw new IllegalArgumentException(
                "Could not pick " + k + " initial centroids — too few points (n=" + n + ").");
        }
        Random rng = new Random(seed);
        double[][] centroids = new double[k][];
        boolean[] taken = new boolean[n];
        int picked = 0;
        while (picked < k) {
            int idx = rng.nextInt(n);
            if (!taken[idx]) {
                taken[idx] = true;
                centroids[picked++] = candidates.get(idx).clone();
            }
        }
        return centroids;
    }

    private static int nearest(double[] p, double[][] centroids, DistanceMetric distance) {
        int best = 0;
        double min = Double.MAX_VALUE;
        for (int i = 0; i < centroids.length; i++) {
            double d = distance.compute(p, centroids[i]);
            if (d < min) {
                min = d;
                best = i;
            }
        }
        return best;
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