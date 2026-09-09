package clustering.algorithms.kmeans;

import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.Geometry;
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

/**
 * Shared centroid-iteration kernel for k-means algorithms.
 * Executes a FLIP-176 feedback loop where each round represents one distributed pass over the dataset.
 * The iteration logic is controlled by a pluggable {@link IterationDriver} running in a parallelism-1 combiner.
 */
public final class CentroidIteration {

    private static final TypeInformation<DenseVector[]> CENTROID_ARRAY_TYPE =
        ObjectArrayTypeInfo.getInfoFor(DenseVectorTypeInfo.INSTANCE);
    private static final TypeInformation<PartialStats> PARTIAL_STATS_TYPE = TypeInformation.of(PartialStats.class);
    private static final TypeInformation<IterationUpdate> ITERATION_UPDATE_TYPE = TypeInformation.of(IterationUpdate.class);

    private CentroidIteration() {}

    /** Decides the next prototype set to evaluate and whether to terminate the iteration. */
    public interface IterationDriver extends Serializable {
        DriverDecision computeNextRound(int epoch, IterationStats stats);
    }

    /** Represents the driver's verdict for a given round. */
    public static final class DriverDecision {
        public final DenseVector[] centroids;
        public final boolean shouldStop;

        public DriverDecision(DenseVector[] centroids, boolean shouldStop) {
            this.centroids = centroids;
            this.shouldStop = shouldStop;
        }

        public static DriverDecision continueIteration(DenseVector[] centroids) {
            return new DriverDecision(centroids, false);
        }

        public static DriverDecision stopIteration(DenseVector[] centroids) {
            return new DriverDecision(centroids, true);
        }
    }

    /** Aggregated statistics for the prototype set during a single distributed pass. */
    public static final class IterationStats {
        public final DenseVector[] centroids;
        public final long[] assignedCounts;
        public final double[] assignedMasses;
        public final double[][] coordinateSums;
        public final double[] clusterErrors;
        public final double[] clusterUtilities;

        IterationStats(DenseVector[] centroids, long[] assignedCounts, double[] assignedMasses, double[][] coordinateSums,
                       double[] clusterErrors, double[] clusterUtilities) {
            this.centroids = centroids;
            this.assignedCounts = assignedCounts;
            this.assignedMasses = assignedMasses;
            this.coordinateSums = coordinateSums;
            this.clusterErrors = clusterErrors;
            this.clusterUtilities = clusterUtilities;
        }

        public double calculateTotalError() {
            double totalError = 0.0;
            for (double error : clusterErrors) {
                totalError += error;
            }
            return totalError;
        }

        public long calculateTotalCount() {
            long totalCount = 0L;
            for (long count : assignedCounts) {
                totalCount += count;
            }
            return totalCount;
        }

        /** Computes the weighted mean of each cell projected onto the provided geometry. */
        public DenseVector[] computeMeans(Geometry geometry) {
            DenseVector[] updatedCentroids = new DenseVector[centroids.length];
            for (int i = 0; i < centroids.length; i++) {
                if (assignedCounts[i] == 0L || assignedMasses[i] == 0.0) {
                    updatedCentroids[i] = centroids[i];
                } else {
                    double[] newMean = new double[coordinateSums[i].length];
                    for (int dim = 0; dim < newMean.length; dim++) {
                        newMean[dim] = coordinateSums[i][dim] / assignedMasses[i];
                    }
                    updatedCentroids[i] = geometry.project(new DenseVector(newMean));
                }
            }
            return updatedCentroids;
        }
    }

    /** Executes the iteration to completion and returns the final centroid set. */
    public static DenseVector[] execute(
            PointSource pointSource,
            EnvFactory envFactory,
            DenseVector[] initialCentroids,
            DistanceMetric distanceMetric,
            IterationDriver driver,
            String jobName) {

        if (initialCentroids.length == 0) {
            throw new IllegalArgumentException("Initial centroids array cannot be empty.");
        }

        StreamExecutionEnvironment env = envFactory.newEnv();
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);

        DataStream<DenseVector[]> initialCentroidStream =
            env.fromCollection(Collections.singletonList(initialCentroids), CENTROID_ARRAY_TYPE);
        DataStream<WeightedPoint> pointsStream = pointSource.create(env);

        DataStreamList iterationResult = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initialCentroidStream),
            ReplayableDataStreamList.notReplay(pointsStream),
            IterationConfig.newBuilder().build(),
            new CentroidIterationBody(distanceMetric, driver, initialCentroids));

        DenseVector[] finalCentroids = FlinkJobs.last(iterationResult.<DenseVector[]>get(0), jobName);
        return finalCentroids == null ? initialCentroids : finalCentroids;
    }

    private static final class CentroidIterationBody implements IterationBody {
        private final DistanceMetric distanceMetric;
        private final IterationDriver driver;
        private final DenseVector[] initialCentroids;

        CentroidIterationBody(DistanceMetric distanceMetric, IterationDriver driver,
                              DenseVector[] initialCentroids) {
            this.distanceMetric = distanceMetric;
            this.driver = driver;
            this.initialCentroids = initialCentroids;
        }

        @Override
        public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<DenseVector[]> centroids = variableStreams.get(0);
            DataStream<WeightedPoint> points = dataStreams.get(0);

            DataStream<PartialStats> partialStats = points
                .connect(centroids.broadcast())
                .transform("centroid-partial-assign", PARTIAL_STATS_TYPE, new PartialAssigner(distanceMetric));

            ManagedMemory.forPointCache(partialStats);

            DataStream<IterationUpdate> updates = partialStats
                .flatMap(new StatsCombiner(driver, initialCentroids))
                .setParallelism(1)
                .returns(ITERATION_UPDATE_TYPE);

            DataStream<DenseVector[]> newCentroids = updates
                .map((MapFunction<IterationUpdate, DenseVector[]>) update -> update.centroids)
                .returns(CENTROID_ARRAY_TYPE)
                .setParallelism(1);

            DataStream<Integer> terminationSignal = updates
                .flatMap(new TerminationEvaluator())
                .returns(Types.INT)
                .setParallelism(1);

            return new IterationBodyResult(
                DataStreamList.of(newCentroids),
                DataStreamList.of(newCentroids),
                terminationSignal);
        }
    }

    /**
     * Parallel per-subtask point assignment. Local points are cached in memory/disk
     * to prevent OOM errors on large datasets.
     */
    private static final class PartialAssigner
            extends AbstractStreamOperator<PartialStats>
            implements TwoInputStreamOperator<WeightedPoint, DenseVector[], PartialStats>,
                       IterationListener<PartialStats> {

        private final DistanceMetric distanceMetric;
        private transient ListStateWithCache<WeightedPoint> cachedPoints;
        private transient ListState<DenseVector[]> currentCentroids;

        PartialAssigner(DistanceMetric distanceMetric) {
            this.distanceMetric = distanceMetric;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            currentCentroids = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("current-centroids", CENTROID_ARRAY_TYPE));
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
        public void processElement2(StreamRecord<DenseVector[]> record) throws Exception {
            currentCentroids.add(record.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<PartialStats> out) throws Exception {
            Optional<DenseVector[]> activeCentroidsOpt = OperatorStateUtils.getUniqueElement(currentCentroids, "current-centroids");
            if (!activeCentroidsOpt.isPresent()) {
                return;
            }
            DenseVector[] activeCentroids = activeCentroidsOpt.get();
            int numCentroids = activeCentroids.length;
            int dimensions = activeCentroids[0].size();

            PartialStats stats = new PartialStats();
            stats.subtaskId = getRuntimeContext().getIndexOfThisSubtask();
            stats.counts = new long[numCentroids];
            stats.masses = new double[numCentroids];
            stats.sums = new double[numCentroids][dimensions];
            stats.errors = new double[numCentroids];
            stats.utilities = new double[numCentroids];

            for (WeightedPoint point : cachedPoints.get()) {
                double[] coordinates = point.features.values;
                double weight = point.weight;

                int closestIndex = 0;
                double closestDist = Double.MAX_VALUE;
                double secondClosestDist = Double.MAX_VALUE;

                for (int i = 0; i < numCentroids; i++) {
                    double dist = distanceMetric.compute(coordinates, activeCentroids[i].values);
                    if (dist < closestDist) {
                        secondClosestDist = closestDist;
                        closestDist = dist;
                        closestIndex = i;
                    } else if (dist < secondClosestDist) {
                        secondClosestDist = dist;
                    }
                }

                double secondClosestDistSq = (secondClosestDist == Double.MAX_VALUE) ? closestDist * closestDist : secondClosestDist * secondClosestDist;
                double[] targetSum = stats.sums[closestIndex];

                for (int dim = 0; dim < dimensions; dim++) {
                    targetSum[dim] += weight * coordinates[dim];
                }

                stats.counts[closestIndex]++;
                stats.masses[closestIndex] += weight;
                stats.errors[closestIndex] += weight * closestDist * closestDist;
                stats.utilities[closestIndex] += weight * (secondClosestDistSq - closestDist * closestDist);
            }
            out.collect(stats);
            currentCentroids.clear();
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

    /** Single-task combiner: merges subtask partials deterministically into global iteration stats. */
    private static final class StatsCombiner
            implements FlatMapFunction<PartialStats, IterationUpdate>, IterationListener<IterationUpdate> {

        private final IterationDriver driver;
        /** The centroids this round's partials were computed against.
         *
         *  <p>Kept here rather than read back off {@code PartialStats}, which used to carry the
         *  whole centroid array from EVERY subtask — k·d doubles per subtask per round (80 KB at
         *  k = 10, d = 1024, so 5 MB per round into this single task at parallelism 64) to re-learn
         *  what this operator emitted itself one round earlier. */
        private final DenseVector[] initialCentroids;
        private transient DenseVector[] currentCentroids;
        private transient List<PartialStats> statsBuffer;

        StatsCombiner(IterationDriver driver, DenseVector[] initialCentroids) {
            this.driver = driver;
            this.initialCentroids = initialCentroids;
        }

        @Override
        public void flatMap(PartialStats partial, Collector<IterationUpdate> out) {
            if (statsBuffer == null) {
                statsBuffer = new ArrayList<>();
            }
            statsBuffer.add(partial);
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<IterationUpdate> out) {
            if (statsBuffer == null || statsBuffer.isEmpty()) {
                return;
            }
            statsBuffer.sort(Comparator.comparingInt(p -> p.subtaskId));
            if (currentCentroids == null) {
                currentCentroids = initialCentroids;
            }
            DenseVector[] centroids = currentCentroids;
            int numCentroids = centroids.length;
            int dimensions = centroids[0].size();

            long[] globalCounts = new long[numCentroids];
            double[] globalMasses = new double[numCentroids];
            double[][] globalSums = new double[numCentroids][dimensions];
            double[] globalErrors = new double[numCentroids];
            double[] globalUtilities = new double[numCentroids];

            for (PartialStats partial : statsBuffer) {
                for (int i = 0; i < numCentroids; i++) {
                    globalCounts[i] += partial.counts[i];
                    globalMasses[i] += partial.masses[i];
                    globalErrors[i] += partial.errors[i];
                    globalUtilities[i] += partial.utilities[i];
                    for (int dim = 0; dim < dimensions; dim++) {
                        globalSums[i][dim] += partial.sums[i][dim];
                    }
                }
            }
            statsBuffer = null;

            DriverDecision decision = driver.computeNextRound(epoch, new IterationStats(centroids, globalCounts, globalMasses, globalSums, globalErrors, globalUtilities));
            currentCentroids = decision.centroids;

            IterationUpdate update = new IterationUpdate();
            update.centroids = decision.centroids;
            update.shouldStop = decision.shouldStop;
            out.collect(update);
        }

        @Override
        public void onIterationTerminated(Context context, Collector<IterationUpdate> out) {}
    }

    /** Emits a continuation token each round unless the round update dictates termination. */
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

    public static final class PartialStats implements Serializable {
        public int subtaskId;
        public long[] counts;
        public double[] masses;
        public double[][] sums;
        public double[] errors;
        public double[] utilities;

        public PartialStats() {}
    }

    public static final class IterationUpdate implements Serializable {
        public DenseVector[] centroids;
        public boolean shouldStop;

        public IterationUpdate() {}
    }

    /**
     * Extracts a seeded sample of target distinct starting centroids directly from the prepared source data.
     */
    public static DenseVector[] sampleInitialCentroids(PointSource preparedSource, EnvFactory envFactory, int targetCentroids, long seed) {
        if (targetCentroids < 1) {
            throw new IllegalArgumentException("Target centroids count must be >= 1, got " + targetCentroids);
        }

        List<WeightedPoint> candidates = Datasets.collectHead(preparedSource, envFactory, Math.max(targetCentroids * 30L, 1000L));
        int numCandidates = candidates.size();

        if (numCandidates < targetCentroids) {
            throw new IllegalArgumentException(
                String.format("Could not sample %d initial centroids — too few points (n=%d).", targetCentroids, numCandidates));
        }

        Random rng = new Random(seed);
        DenseVector[] sampledCentroids = new DenseVector[targetCentroids];
        boolean[] selectedIndices = new boolean[numCandidates];
        int pointsPicked = 0;

        while (pointsPicked < targetCentroids) {
            int candidateIndex = rng.nextInt(numCandidates);
            if (!selectedIndices[candidateIndex]) {
                selectedIndices[candidateIndex] = true;
                sampledCentroids[pointsPicked++] = new DenseVector(candidates.get(candidateIndex).features.values.clone());
            }
        }
        return sampledCentroids;
    }

    /** Calculates the maximum movement distance between two equally-sized centroid sets. */
    public static double calculateMaxMovement(DenseVector[] previousCentroids, DenseVector[] currentCentroids, DistanceMetric distanceMetric) {
        double maxMovement = 0.0;
        for (int i = 0; i < previousCentroids.length; i++) {
            maxMovement = Math.max(maxMovement, distanceMetric.compute(previousCentroids[i], currentCentroids[i]));
        }
        return maxMovement;
    }
}
