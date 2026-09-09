package clustering.algorithms.kmeans.hierarchical;

import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import clustering.core.Clusterer;
import clustering.core.EnvFactory;
import clustering.core.EuclideanGeometry;
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
import java.util.Random;
import java.util.Set;

/**
 * Bisecting k-means implementation.
 * Divisively splits the highest-cost or largest leaf using 2-means until k leaves exist.
 * The entire search runs as a single Flink job to avoid materializing intermediate subsets.
 */
public class BisectingKMeans implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(BisectingKMeans.class);

    private static final TypeInformation<IterationState> ITERATION_STATE_TYPE = TypeInformation.of(IterationState.class);
    private static final TypeInformation<PartialStats> PARTIAL_STATS_TYPE = TypeInformation.of(PartialStats.class);
    private static final TypeInformation<IterationUpdate> ITERATION_UPDATE_TYPE = TypeInformation.of(IterationUpdate.class);

    private static final int SEED_SAMPLE_CAP = 256;

    static final int PHASE_INIT_GLOBAL_MEAN = 0;
    static final int PHASE_SEED = 1;
    static final int PHASE_LLOYD = 2;
    static final int PHASE_COMMIT = 3;

    private final int targetK;
    private final int maxIterations;
    private final double tolerance;
    private final long seed;
    private final Geometry geometry;
    private final int numTrials;
    private final String selectionStrategy;

    public BisectingKMeans(int targetK, int maxIterations, double tolerance, long seed, Geometry geometry,
                           int numTrials, String selectionStrategy) {
        if (numTrials < 1) {
            throw new IllegalArgumentException("trials must be >= 1, got " + numTrials);
        }
        String mode = selectionStrategy.toLowerCase();
        if (!mode.equals("cost") && !mode.equals("size")) {
            throw new IllegalArgumentException(
                "Unknown select mode: '" + selectionStrategy + "'. Known: cost, size");
        }
        this.targetK = targetK;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.seed = seed;
        this.geometry = geometry;
        this.numTrials = numTrials;
        this.selectionStrategy = mode;
    }

    public BisectingKMeans(int targetK, int maxIterations, double tolerance, long seed, Geometry geometry) {
        this(targetK, maxIterations, tolerance, seed, geometry, 1, "cost");
    }

    public BisectingKMeans(int targetK, int maxIterations, double tolerance, long seed) {
        this(targetK, maxIterations, tolerance, seed, EuclideanGeometry.INSTANCE);
    }

    @Override
    public BisectingKMeansModel fit(PointSource source, EnvFactory envFactory, int parallelism) {
        PointSource preparedSource = geometry.prepare(source);

        StreamExecutionEnvironment env = envFactory.newEnv();
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);

        DataStream<IterationState> initialStateStream =
            env.fromCollection(Collections.singletonList(IterationState.initialize(targetK)), ITERATION_STATE_TYPE);
        DataStream<WeightedPoint> pointsStream = preparedSource.create(env);

        DataStreamList iterationResult = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initialStateStream),
            ReplayableDataStreamList.notReplay(pointsStream),
            IterationConfig.newBuilder().build(),
            new BisectingIterationBody(targetK, maxIterations, tolerance, geometry, seed, numTrials, selectionStrategy));

        IterationState finalState = FlinkJobs.last(iterationResult.<IterationState>get(0), "bisectingkmeans-fit");
        if (finalState == null || finalState.totalNodes == 0) {
            throw new RuntimeException("Bisecting k-means produced no cluster tree");
        }
        return new BisectingKMeansModel(
            Arrays.copyOf(finalState.centroids, finalState.totalNodes),
            Arrays.copyOf(finalState.leftChildren, finalState.totalNodes),
            Arrays.copyOf(finalState.rightChildren, finalState.totalNodes),
            Arrays.copyOf(finalState.assignedLeafIds, finalState.totalNodes),
            geometry.modelDistance());
    }

    private static final class BisectingIterationBody implements IterationBody {
        private final int targetK;
        private final int maxIterations;
        private final double tolerance;
        private final Geometry geometry;
        private final long seed;
        private final int numTrials;
        private final String selectionStrategy;

        BisectingIterationBody(int targetK, int maxIterations, double tolerance, Geometry geometry, long seed,
                               int numTrials, String selectionStrategy) {
            this.targetK = targetK;
            this.maxIterations = maxIterations;
            this.tolerance = tolerance;
            this.geometry = geometry;
            this.seed = seed;
            this.numTrials = numTrials;
            this.selectionStrategy = selectionStrategy;
        }

        @Override
        public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<IterationState> state = variableStreams.get(0);
            DataStream<WeightedPoint> points = dataStreams.get(0);

            DataStream<PartialStats> partialStats = points
                .connect(state.broadcast())
                .transform("bisecting-partial", PARTIAL_STATS_TYPE, new PartitionEvaluator(geometry.fitDistance(), seed));

            ManagedMemory.forPointCache(partialStats);

            DataStream<IterationUpdate> updates = partialStats
                .flatMap(new StatsCombiner(targetK, maxIterations, tolerance, geometry, seed, numTrials, selectionStrategy))
                .setParallelism(1)
                .returns(ITERATION_UPDATE_TYPE);

            DataStream<IterationState> newState = updates
                .map((MapFunction<IterationUpdate, IterationState>) update -> update.state)
                .returns(ITERATION_STATE_TYPE)
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
     * Parallel per-subtask fold. Caches points once and computes local statistics
     * based on the current active node in the tree.
     */
    private static final class PartitionEvaluator
            extends AbstractStreamOperator<PartialStats>
            implements TwoInputStreamOperator<WeightedPoint, IterationState, PartialStats>,
                       IterationListener<PartialStats> {

        private final DistanceMetric distanceMetric;
        private final long seed;
        private transient ListStateWithCache<WeightedPoint> cachedPoints;
        private transient ListState<IterationState> stateList;

        PartitionEvaluator(DistanceMetric distanceMetric, long seed) {
            this.distanceMetric = distanceMetric;
            this.seed = seed;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            stateList = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("bisecting-state", ITERATION_STATE_TYPE));
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
            Optional<IterationState> currentStateOpt = OperatorStateUtils.getUniqueElement(stateList, "bisecting-state");
            if (!currentStateOpt.isPresent()) {
                return;
            }
            IterationState state = currentStateOpt.get();
            PartialStats stats = new PartialStats();
            stats.subtaskId = getRuntimeContext().getIndexOfThisSubtask();
            stats.state = state;
            stats.branchCounts = new long[2];
            stats.branchMasses = new double[2];
            stats.branchCosts = new double[2];

            if (state.phase == PHASE_INIT_GLOBAL_MEAN) {
                aggregateGlobalMean(stats);
            } else if (state.phase == PHASE_SEED) {
                collectSeedSample(state, stats);
            } else {
                evaluateSplitBranches(state, stats);
            }

            out.collect(stats);
            stateList.clear();
        }

        private void aggregateGlobalMean(PartialStats stats) throws Exception {
            for (WeightedPoint point : cachedPoints.get()) {
                double[] coords = point.features.values;
                if (stats.branchSums == null) {
                    stats.branchSums = new double[2][coords.length];
                }
                addScaledVector(stats.branchSums[0], coords, point.weight);
                stats.branchCounts[0]++;
                stats.branchMasses[0] += point.weight;
            }
        }

        private void collectSeedSample(IterationState state, PartialStats stats) throws Exception {
            List<DenseVector> reservoir = new ArrayList<>(SEED_SAMPLE_CAP);
            Random rng = new Random(seed
                + 1_000_003L * (state.targetNodeId + 1)
                + 31L * (state.currentTrial + 1)
                + 131L * getRuntimeContext().getIndexOfThisSubtask());

            long pointsSeen = 0L;
            for (WeightedPoint point : cachedPoints.get()) {
                DenseVector coords = point.features;
                if (findLeafNode(coords, state, distanceMetric) != state.targetNodeId) {
                    continue;
                }
                stats.branchCounts[0]++;
                stats.branchMasses[0] += point.weight;

                if (reservoir.size() < SEED_SAMPLE_CAP) {
                    reservoir.add(new DenseVector(coords.values.clone()));
                } else {
                    long slot = (long) (rng.nextDouble() * (pointsSeen + 1));
                    if (slot < SEED_SAMPLE_CAP) {
                        reservoir.set((int) slot, new DenseVector(coords.values.clone()));
                    }
                }
                pointsSeen++;
            }
            stats.sample = reservoir.toArray(new DenseVector[0]);
        }

        private void evaluateSplitBranches(IterationState state, PartialStats stats) throws Exception {
            DenseVector[] splitCentroids = state.activeSplitCentroids;
            for (WeightedPoint point : cachedPoints.get()) {
                DenseVector coords = point.features;
                if (findLeafNode(coords, state, distanceMetric) != state.targetNodeId) {
                    continue;
                }
                if (stats.branchSums == null) {
                    stats.branchSums = new double[2][coords.size()];
                }
                double weight = point.weight;
                double dist0 = distanceMetric.compute(coords, splitCentroids[0]);
                double dist1 = distanceMetric.compute(coords, splitCentroids[1]);

                int branch = dist0 <= dist1 ? 0 : 1;
                double dist = branch == 0 ? dist0 : dist1;

                addScaledVector(stats.branchSums[branch], coords.values, weight);
                stats.branchCounts[branch]++;
                stats.branchMasses[branch] += weight;
                stats.branchCosts[branch] += weight * dist * dist;
            }
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
     * Single-task combiner: sums the round's partials and orchestrates the split search.
     */
    private static final class StatsCombiner
            implements FlatMapFunction<PartialStats, IterationUpdate>, IterationListener<IterationUpdate> {

        private final int targetK;
        private final int maxIterations;
        private final double tolerance;
        private final Geometry geometry;
        private final long seed;
        private final int numTrials;
        private final boolean selectBySize;

        private transient List<PartialStats> statsBuffer;
        private transient Map<Integer, Double> leafCosts;
        private transient Map<Integer, Long> leafCounts;
        private transient Map<Integer, Double> leafMasses;
        private transient Set<Integer> unsplittableLeaves;
        private transient int currentSplitRounds;

        StatsCombiner(int targetK, int maxIterations, double tolerance, Geometry geometry, long seed, int numTrials,
                      String selectionStrategy) {
            this.targetK = targetK;
            this.maxIterations = maxIterations;
            this.tolerance = tolerance;
            this.geometry = geometry;
            this.seed = seed;
            this.numTrials = numTrials;
            this.selectBySize = selectionStrategy.equals("size");
        }

        private double calculateLeafScore(int nodeId) {
            return selectBySize
                ? leafMasses.getOrDefault(nodeId, 0.0)
                : leafCosts.getOrDefault(nodeId, Double.NEGATIVE_INFINITY);
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
            if (leafCosts == null) {
                leafCosts = new HashMap<>();
                leafCounts = new HashMap<>();
                leafMasses = new HashMap<>();
                unsplittableLeaves = new HashSet<>();
            }
            statsBuffer.sort(Comparator.comparingInt(p -> p.subtaskId));
            IterationState state = statsBuffer.get(0).state;

            long[] globalCounts = new long[2];
            double[] globalMasses = new double[2];
            double[] globalCosts = new double[2];
            double[][] globalSums = null;
            List<DenseVector> aggregatedSample = new ArrayList<>();

            for (PartialStats partial : statsBuffer) {
                globalCounts[0] += partial.branchCounts[0];
                globalCounts[1] += partial.branchCounts[1];
                globalMasses[0] += partial.branchMasses[0];
                globalMasses[1] += partial.branchMasses[1];
                globalCosts[0] += partial.branchCosts[0];
                globalCosts[1] += partial.branchCosts[1];

                if (partial.branchSums != null) {
                    if (globalSums == null) {
                        globalSums = new double[2][partial.branchSums[0].length];
                    }
                    addScaledVector(globalSums[0], partial.branchSums[0], 1.0);
                    addScaledVector(globalSums[1], partial.branchSums[1], 1.0);
                }
                if (partial.sample != null) {
                    Collections.addAll(aggregatedSample, partial.sample);
                }
            }
            statsBuffer = null;

            // Pinned by content, not by which subtask a rebalance channel happened to route a
            // row to: the reservoir per subtask is already an unbiased sample of ITS points, but
            // concatenating subtasks in id order still leaves the MERGED list's order dependent on
            // Flink's random rebalance start channel. Sorting here is free (single combiner task,
            // <= SEED_SAMPLE_CAP-ish rows) and makes both the trim below and trial 0's "first" seed
            // (extractSeeds's sample.get(0)) a pure function of the sampled point SET, not of which
            // physical channel a row landed on.
            aggregatedSample.sort(BisectingKMeans::compareByCoordinates);

            if (aggregatedSample.size() > SEED_SAMPLE_CAP) {
                Collections.shuffle(aggregatedSample, new Random(
                    seed + 1_000_003L * (state.targetNodeId + 1) + 31L * (state.currentTrial + 1)));
                aggregatedSample.subList(SEED_SAMPLE_CAP, aggregatedSample.size()).clear();
            }

            IterationUpdate update;
            switch (state.phase) {
                case PHASE_INIT_GLOBAL_MEAN:
                    update = initializeRootNode(state, globalCounts[0], globalMasses[0], globalSums);
                    break;
                case PHASE_SEED:
                    update = initiateSplit(state, globalCounts[0], globalMasses[0], aggregatedSample);
                    break;
                case PHASE_LLOYD:
                    update = executeLloydIteration(state, globalMasses, globalSums);
                    break;
                default:
                    update = evaluateTrialResult(state, globalCounts, globalMasses, globalCosts);
                    break;
            }
            out.collect(update);
        }

        private IterationUpdate initializeRootNode(IterationState state, long count, double mass, double[][] sums) {
            if (count < targetK) {
                throw new IllegalStateException("Dataset too small: n=" + count + " rows but k=" + targetK + " requested.");
            }
            IterationState nextState = state.copy();
            int rootNode = nextState.allocateNodes(1);
            nextState.centroids[rootNode] = geometry.project(calculateMean(sums[0], mass));
            leafCounts.put(0, count);
            leafMasses.put(0, mass);
            leafCosts.put(0, Double.MAX_VALUE);

            if (targetK == 1) {
                return terminateIteration(nextState);
            }
            return prepareNextLeaf(nextState, 0);
        }

        private IterationUpdate prepareNextLeaf(IterationState state, int leafId) {
            state.phase = PHASE_SEED;
            state.targetNodeId = leafId;
            state.currentTrial = 0;
            state.activeSplitCentroids = null;
            state.bestSplitCentroids = null;
            state.minSplitCost = Double.MAX_VALUE;
            state.bestBranchCounts = null;
            state.bestBranchCosts = null;
            return continueIteration(state);
        }

        private IterationUpdate initiateSplit(IterationState state, long targetRows, double targetMass,
                                              List<DenseVector> sample) {
            IterationState nextState = state.copy();
            leafCounts.put(state.targetNodeId, targetRows);
            leafMasses.put(state.targetNodeId, targetMass);

            if (sample.size() < 2) {
                return processSettledLeaf(nextState, state.targetNodeId, true);
            }

            DenseVector[] seeds = extractSeeds(sample, state.targetNodeId, state.currentTrial);
            if (seeds == null) {
                return processSettledLeaf(nextState, state.targetNodeId, true);
            }

            nextState.activeSplitCentroids = seeds;
            nextState.phase = PHASE_LLOYD;
            currentSplitRounds = 0;
            return continueIteration(nextState);
        }

        private DenseVector[] extractSeeds(List<DenseVector> sample, int leafId, int trialNum) {
            DistanceMetric metric = geometry.fitDistance();
            int startIndex = trialNum == 0 ? 0 :
                new Random(seed + 1_000_003L * (leafId + 1) + 31L * trialNum).nextInt(sample.size());

            DenseVector firstSeed = sample.get(startIndex);
            DenseVector farthestSeed = null;
            double maxDist = 0.0;

            for (DenseVector point : sample) {
                double dist = metric.compute(firstSeed, point);
                if (dist > maxDist) {
                    maxDist = dist;
                    farthestSeed = point;
                }
            }

            return farthestSeed == null ? null : new DenseVector[] {
                new DenseVector(firstSeed.values.clone()), new DenseVector(farthestSeed.values.clone())};
        }

        private IterationUpdate executeLloydIteration(IterationState state, double[] masses, double[][] sums) {
            IterationState nextState = state.copy();
            DenseVector[] newCentroids = new DenseVector[2];

            for (int i = 0; i < 2; i++) {
                newCentroids[i] = masses[i] == 0.0 || sums == null
                    ? state.activeSplitCentroids[i]
                    : geometry.project(calculateMean(sums[i], masses[i]));
            }

            double maxMovement = Math.max(
                geometry.fitDistance().compute(state.activeSplitCentroids[0], newCentroids[0]),
                geometry.fitDistance().compute(state.activeSplitCentroids[1], newCentroids[1]));

            nextState.activeSplitCentroids = newCentroids;
            currentSplitRounds++;

            if (maxMovement < tolerance || currentSplitRounds >= maxIterations) {
                nextState.phase = PHASE_COMMIT;
            }
            return continueIteration(nextState);
        }

        private IterationUpdate evaluateTrialResult(IterationState state, long[] counts, double[] masses, double[] costs) {
            IterationState nextState = state.copy();

            boolean isDegenerate = counts[0] == 0L || counts[1] == 0L;
            if (!isDegenerate) {
                double totalCost = costs[0] + costs[1];
                if (totalCost < nextState.minSplitCost) {
                    nextState.minSplitCost = totalCost;
                    nextState.bestSplitCentroids = state.activeSplitCentroids;
                    nextState.bestBranchCounts = counts.clone();
                    nextState.bestBranchMasses = masses.clone();
                    nextState.bestBranchCosts = costs.clone();
                }
            }

            int nextTrialId = state.currentTrial + 1;
            if (nextTrialId < numTrials) {
                nextState.currentTrial = nextTrialId;
                nextState.phase = PHASE_SEED;
                nextState.activeSplitCentroids = null;
                return continueIteration(nextState);
            }

            if (nextState.bestSplitCentroids == null) {
                logger.warn("bisecting k-means: all {} trial(s) splitting node {} produced an empty child; marking unsplittable",
                    numTrials, state.targetNodeId);
                return processSettledLeaf(nextState, state.targetNodeId, true);
            }
            return finalizeSplit(nextState, state.targetNodeId);
        }

        private IterationUpdate finalizeSplit(IterationState state, int targetNodeId) {
            int leftChildId = state.allocateNodes(2);
            int rightChildId = leftChildId + 1;

            state.centroids[leftChildId] = state.bestSplitCentroids[0];
            state.centroids[rightChildId] = state.bestSplitCentroids[1];
            state.leftChildren[targetNodeId] = leftChildId;
            state.rightChildren[targetNodeId] = rightChildId;

            leafCosts.remove(targetNodeId);
            leafCounts.remove(targetNodeId);
            leafMasses.remove(targetNodeId);

            leafCosts.put(leftChildId, state.bestBranchCosts[0]);
            leafCosts.put(rightChildId, state.bestBranchCosts[1]);
            leafCounts.put(leftChildId, state.bestBranchCounts[0]);
            leafCounts.put(rightChildId, state.bestBranchCounts[1]);
            leafMasses.put(leftChildId, state.bestBranchMasses[0]);
            leafMasses.put(rightChildId, state.bestBranchMasses[1]);

            if (numTrials > 1) {
                logger.debug("bisecting k-means: leaf {} split on best of {} trials, childCost={}",
                    targetNodeId, numTrials, state.minSplitCost);
            }
            return processSettledLeaf(state, -1, false);
        }

        private IterationUpdate processSettledLeaf(IterationState state, int settledLeafId, boolean markUnsplittable) {
            if (markUnsplittable) {
                unsplittableLeaves.add(settledLeafId);
                state.activeSplitCentroids = null;
            }
            if (leafCosts.size() >= targetK) {
                return terminateIteration(state);
            }

            int nextTargetId = -1;
            double highestScore = Double.NEGATIVE_INFINITY;

            for (Integer nodeId : leafCosts.keySet()) {
                boolean canSplit = !unsplittableLeaves.contains(nodeId)
                    && leafCounts.getOrDefault(nodeId, 0L) >= 2L
                    && leafCosts.getOrDefault(nodeId, 0.0) > 0.0;

                if (!canSplit) {
                    continue;
                }
                double score = calculateLeafScore(nodeId);
                if (score > highestScore || (score == highestScore && nodeId < nextTargetId)) {
                    highestScore = score;
                    nextTargetId = nodeId;
                }
            }

            if (nextTargetId < 0) {
                logger.warn("bisecting k-means: no splittable leaf left after {} clusters (requested k={}) — stopping early",
                    leafCosts.size(), targetK);
                return terminateIteration(state);
            }
            return prepareNextLeaf(state, nextTargetId);
        }

        private IterationUpdate terminateIteration(IterationState state) {
            state.assignedLeafIds = new int[state.centroids.length];
            Arrays.fill(state.assignedLeafIds, -1);
            assignLeafIds(state, 0, new int[] {0});
            IterationUpdate update = new IterationUpdate();
            update.state = state;
            update.shouldStop = true;
            return update;
        }

        private static void assignLeafIds(IterationState state, int nodeId, int[] nextId) {
            if (state.leftChildren[nodeId] < 0) {
                state.assignedLeafIds[nodeId] = nextId[0]++;
                return;
            }
            assignLeafIds(state, state.leftChildren[nodeId], nextId);
            assignLeafIds(state, state.rightChildren[nodeId], nextId);
        }

        private static IterationUpdate continueIteration(IterationState state) {
            IterationUpdate update = new IterationUpdate();
            update.state = state;
            update.shouldStop = false;
            return update;
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
        public int phase;
        public DenseVector[] centroids;
        public int[] leftChildren;
        public int[] rightChildren;
        public int totalNodes;
        public int targetNodeId;
        public DenseVector[] activeSplitCentroids;
        public int currentTrial;
        public DenseVector[] bestSplitCentroids;
        public double minSplitCost;
        public long[] bestBranchCounts;
        public double[] bestBranchMasses;
        public double[] bestBranchCosts;
        public int[] assignedLeafIds;

        public IterationState() {}

        static IterationState initialize(int k) {
            IterationState state = new IterationState();
            state.phase = PHASE_INIT_GLOBAL_MEAN;
            state.centroids = new DenseVector[0];
            state.leftChildren = new int[0];
            state.rightChildren = new int[0];
            state.totalNodes = 0;
            state.targetNodeId = -1;
            state.currentTrial = 0;
            state.minSplitCost = Double.MAX_VALUE;
            return state;
        }

        int allocateNodes(int numNewNodes) {
            int firstNewIndex = totalNodes;
            centroids = Arrays.copyOf(centroids, totalNodes + numNewNodes);
            leftChildren = Arrays.copyOf(leftChildren, totalNodes + numNewNodes);
            rightChildren = Arrays.copyOf(rightChildren, totalNodes + numNewNodes);
            for (int i = firstNewIndex; i < firstNewIndex + numNewNodes; i++) {
                leftChildren[i] = -1;
                rightChildren[i] = -1;
            }
            totalNodes += numNewNodes;
            return firstNewIndex;
        }

        IterationState copy() {
            IterationState copy = new IterationState();
            copy.phase = phase;
            copy.centroids = centroids.clone();
            copy.leftChildren = leftChildren.clone();
            copy.rightChildren = rightChildren.clone();
            copy.totalNodes = totalNodes;
            copy.targetNodeId = targetNodeId;
            copy.activeSplitCentroids = activeSplitCentroids;
            copy.currentTrial = currentTrial;
            copy.bestSplitCentroids = bestSplitCentroids;
            copy.minSplitCost = minSplitCost;
            copy.bestBranchCounts = bestBranchCounts;
            copy.bestBranchMasses = bestBranchMasses;
            copy.bestBranchCosts = bestBranchCosts;
            copy.assignedLeafIds = assignedLeafIds;
            return copy;
        }
    }

    public static final class PartialStats implements Serializable {
        public int subtaskId;
        public long[] branchCounts;
        public double[] branchMasses;
        public double[][] branchSums;
        public double[] branchCosts;
        public DenseVector[] sample;
        public IterationState state;

        public PartialStats() {}
    }

    public static final class IterationUpdate implements Serializable {
        public IterationState state;
        public boolean shouldStop;

        public IterationUpdate() {}
    }

    static int findLeafNode(DenseVector point, IterationState state, DistanceMetric distanceMetric) {
        int nodeId = 0;
        while (state.leftChildren[nodeId] >= 0) {
            int leftId = state.leftChildren[nodeId];
            int rightId = state.rightChildren[nodeId];
            nodeId = distanceMetric.compute(point, state.centroids[leftId]) <= distanceMetric.compute(point, state.centroids[rightId]) ? leftId : rightId;
        }
        return nodeId;
    }

    private static void addScaledVector(double[] accumulator, double[] point, double weight) {
        for (int i = 0; i < accumulator.length; i++) {
            accumulator[i] += weight * point[i];
        }
    }

    private static int compareByCoordinates(DenseVector a, DenseVector b) {
        double[] av = a.values;
        double[] bv = b.values;
        int shared = Math.min(av.length, bv.length);
        for (int i = 0; i < shared; i++) {
            int c = Double.compare(av[i], bv[i]);
            if (c != 0) {
                return c;
            }
        }
        return Integer.compare(av.length, bv.length);
    }

    private static DenseVector calculateMean(double[] sumVector, double totalMass) {
        double[] meanVector = new double[sumVector.length];
        for (int i = 0; i < sumVector.length; i++) {
            meanVector[i] = sumVector[i] / totalMass;
        }
        return new DenseVector(meanVector);
    }
}
