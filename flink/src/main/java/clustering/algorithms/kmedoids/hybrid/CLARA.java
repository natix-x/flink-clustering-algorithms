package clustering.algorithms.kmedoids.hybrid;

import clustering.algorithms.kmedoids.KMedoidsModel;
import clustering.algorithms.kmedoids.components.DriverSample;
import clustering.algorithms.kmedoids.local.DriverLocalKMedoids;
import clustering.core.Clusterer;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.ManagedMemory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.core.Points;
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
import org.apache.flink.ml.linalg.DenseVector;
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
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Random;

/**
 * CLARA (Clustering LARge Applications) implementation.
 * Uses a Flink ML bounded iteration to draw random samples in the first round,
 * solve them locally, and score them against the full dataset in the second round.
 */
public class CLARA implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(CLARA.class);

    private static final TypeInformation<IterationState> STATE_TYPE = TypeInformation.of(IterationState.class);
    private static final TypeInformation<PartialStats> PARTIAL_TYPE = TypeInformation.of(PartialStats.class);
    private static final TypeInformation<IterationUpdate> UPDATE_TYPE = TypeInformation.of(IterationUpdate.class);

    private static final int PHASE_SAMPLE = 0;
    private static final int PHASE_EVALUATE = 1;

    private final int targetK;
    private final int numSamples;
    private final int sampleSize;
    private final DistanceMetric distanceMetric;
    private final long seed;
    private final String localSolverName;
    private final DriverLocalKMedoids localSolver;

    public CLARA(int targetK, int numSamples, int sampleSize, int maxIterations, DistanceMetric distanceMetric,
                 String localSolverName, long seed) {
        this.targetK = targetK;
        this.numSamples = numSamples;
        this.sampleSize = sampleSize;
        this.distanceMetric = distanceMetric;
        this.seed = seed;
        this.localSolverName = localSolverName;
        this.localSolver = DriverLocalKMedoids.fromName(localSolverName, targetK, maxIterations, distanceMetric, seed);
    }

    @Override
    public Model fit(PointSource source, EnvFactory envFactory, int parallelism) {
        return initializeMedoids(source, envFactory).model;
    }

    /** Prepares the initial medoids by running the CLARA sampling and scoring phases. */
    Seeding initializeMedoids(PointSource source, EnvFactory envFactory) {
        long totalPoints = Datasets.count(source, envFactory);
        if (totalPoints < targetK) {
            throw new IllegalArgumentException("Dataset too small: n=" + totalPoints + " points but k=" + targetK + " requested.");
        }

        if (totalPoints <= sampleSize) {
            List<WeightedPoint> allPoints = Datasets.collectAll(source, envFactory);
            KMedoidsModel model = localSolver.fitLocal(Points.toArray(allPoints), Points.weightsOf(allPoints));
            double cost = calculateTotalCost(model.medoids(), allPoints);
            logger.info("clara: n={} samples=1 (whole dataset) solver={} cost={}", totalPoints, localSolverName, cost);
            return new Seeding(model, cost, totalPoints);
        }

        double samplingFraction = DriverSample.calculateInclusionProbability(sampleSize, totalPoints);

        StreamExecutionEnvironment env = envFactory.newEnv();
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);

        DataStream<IterationState> initStateStream =
            env.fromCollection(Collections.singletonList(IterationState.initialize()), STATE_TYPE);
        DataStream<WeightedPoint> pointsStream = source.create(env);

        DataStreamList iterationResult = Iterations.iterateBoundedStreamsUntilTermination(
            DataStreamList.of(initStateStream),
            ReplayableDataStreamList.notReplay(pointsStream),
            IterationConfig.newBuilder().build(),
            new ClaraIterationBody(targetK, numSamples, sampleSize, samplingFraction, distanceMetric, seed, localSolver));

        IterationState finalState = FlinkJobs.last(iterationResult.<IterationState>get(0), "clara-fit");
        if (finalState == null || finalState.bestMedoidSet == null) {
            throw new IllegalStateException("CLARA produced no medoids (n=" + totalPoints + ")");
        }

        DenseVector[] finalMedoids = new DenseVector[finalState.bestMedoidSet.length];
        for (int i = 0; i < finalMedoids.length; i++) {
            finalMedoids[i] = Points.wrap(finalState.bestMedoidSet[i]);
        }
        logger.info("clara: n={} samples={} sampleSize={} solver={} cost={}",
            totalPoints, numSamples, sampleSize, localSolverName, finalState.minCost);
        return new Seeding(new KMedoidsModel(finalMedoids, distanceMetric), finalState.minCost, totalPoints);
    }

    private double calculateTotalCost(DenseVector[] medoids, List<WeightedPoint> points) {
        double totalCost = 0.0;
        for (WeightedPoint point : points) {
            double[] coords = point.features.values;
            double minDist = Double.MAX_VALUE;
            for (DenseVector medoid : medoids) {
                double dist = distanceMetric.compute(medoid.values, coords);
                if (dist < minDist) {
                    minDist = dist;
                }
            }
            totalCost += point.weight * minDist;
        }
        return totalCost;
    }

    static final class Seeding {
        final KMedoidsModel model;
        final double cost;
        final long totalPoints;

        Seeding(KMedoidsModel model, double cost, long totalPoints) {
            this.model = model;
            this.cost = cost;
            this.totalPoints = totalPoints;
        }
    }

    private static final class ClaraIterationBody implements IterationBody {
        private final int targetK;
        private final int numSamples;
        private final int sampleSize;
        private final double samplingFraction;
        private final DistanceMetric distanceMetric;
        private final long seed;
        private final DriverLocalKMedoids localSolver;

        ClaraIterationBody(int targetK, int numSamples, int sampleSize, double samplingFraction,
                           DistanceMetric distanceMetric, long seed, DriverLocalKMedoids localSolver) {
            this.targetK = targetK;
            this.numSamples = numSamples;
            this.sampleSize = sampleSize;
            this.samplingFraction = samplingFraction;
            this.distanceMetric = distanceMetric;
            this.seed = seed;
            this.localSolver = localSolver;
        }

        @Override
        public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<IterationState> stateStream = variableStreams.get(0);
            DataStream<WeightedPoint> pointsStream = dataStreams.get(0);

            DataStream<PartialStats> partialStats = pointsStream
                .connect(stateStream.broadcast())
                .transform("clara-fold", PARTIAL_TYPE,
                    new PhaseEvaluator(numSamples, samplingFraction, distanceMetric, seed));

            ManagedMemory.forPointCache(partialStats);

            DataStream<IterationUpdate> updates = partialStats
                .flatMap(new PhaseCombiner(targetK, numSamples, sampleSize, seed, localSolver))
                .setParallelism(1)
                .returns(UPDATE_TYPE);

            DataStream<IterationState> newState = updates
                .map((MapFunction<IterationUpdate, IterationState>) update -> update.state)
                .returns(STATE_TYPE)
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

    /** Subtask evaluator: draws local samples in phase 0, computes partial costs in phase 1. */
    private static final class PhaseEvaluator
            extends AbstractStreamOperator<PartialStats>
            implements TwoInputStreamOperator<WeightedPoint, IterationState, PartialStats>,
                       IterationListener<PartialStats> {

        private final int numSamples;
        private final double samplingFraction;
        private final DistanceMetric distanceMetric;
        private final long seed;
        private transient ListStateWithCache<WeightedPoint> cachedPoints;
        private transient ListState<IterationState> iterationStateList;

        PhaseEvaluator(int numSamples, double samplingFraction, DistanceMetric distanceMetric, long seed) {
            this.numSamples = numSamples;
            this.samplingFraction = samplingFraction;
            this.distanceMetric = distanceMetric;
            this.seed = seed;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            iterationStateList = context.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("clara-state", STATE_TYPE));
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
            iterationStateList.add(record.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(int epoch, Context context, Collector<PartialStats> out) throws Exception {
            Optional<IterationState> currentStateOpt = OperatorStateUtils.getUniqueElement(iterationStateList, "clara-state");
            if (!currentStateOpt.isPresent()) {
                return;
            }
            IterationState state = currentStateOpt.get();
            PartialStats stats = new PartialStats();
            stats.subtaskId = getRuntimeContext().getIndexOfThisSubtask();
            stats.state = state;

            if (state.phase == PHASE_SAMPLE) {
                drawSamples(stats);
            } else {
                stats.candidateCosts = evaluateCandidateCosts(state.candidateSets, state.targetK);
            }
            out.collect(stats);
            iterationStateList.clear();
        }

        private void drawSamples(PartialStats stats) throws Exception {
            Random randomGen = new Random(seed + 31L * getRuntimeContext().getIndexOfThisSubtask());
            List<double[]> sampledCoords = new ArrayList<>();
            List<Double> sampledWeights = new ArrayList<>();
            List<Integer> sampleIndices = new ArrayList<>();

            for (WeightedPoint point : cachedPoints.get()) {
                double[] coords = point.features.values;
                double weight = point.weight;
                for (int s = 0; s < numSamples; s++) {
                    if (randomGen.nextDouble() < samplingFraction) {
                        sampledCoords.add(coords.clone());
                        sampledWeights.add(weight);
                        sampleIndices.add(s);
                    }
                }
            }

            stats.sampledCoords = sampledCoords.toArray(new double[0][]);
            stats.sampledWeights = new double[sampledWeights.size()];
            stats.sampleIndices = new int[sampleIndices.size()];
            for (int i = 0; i < stats.sampledWeights.length; i++) {
                stats.sampledWeights[i] = sampledWeights.get(i);
                stats.sampleIndices[i] = sampleIndices.get(i);
            }
        }

        private double[] evaluateCandidateCosts(double[][] candidateSets, int targetK) throws Exception {
            int numSets = candidateSets.length / targetK;
            double[] costs = new double[numSets];

            for (WeightedPoint point : cachedPoints.get()) {
                double[] coords = point.features.values;
                double weight = point.weight;

                for (int setIdx = 0; setIdx < numSets; setIdx++) {
                    double minDist = Double.MAX_VALUE;
                    int baseIdx = setIdx * targetK;
                    for (int i = 0; i < targetK; i++) {
                        double dist = distanceMetric.compute(candidateSets[baseIdx + i], coords);
                        if (dist < minDist) {
                            minDist = dist;
                        }
                    }
                    costs[setIdx] += weight * minDist;
                }
            }
            return costs;
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

    /** Combiner: assembles samples and runs local solver (phase 0) or selects the best medoid set (phase 1). */
    private static final class PhaseCombiner
            implements FlatMapFunction<PartialStats, IterationUpdate>, IterationListener<IterationUpdate> {

        private final int targetK;
        private final int numSamples;
        private final int sampleSize;
        private final long seed;
        private final DriverLocalKMedoids localSolver;
        private transient List<PartialStats> statsBuffer;

        PhaseCombiner(int targetK, int numSamples, int sampleSize, long seed, DriverLocalKMedoids localSolver) {
            this.targetK = targetK;
            this.numSamples = numSamples;
            this.sampleSize = sampleSize;
            this.seed = seed;
            this.localSolver = localSolver;
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
            statsBuffer.sort(Comparator.comparingInt(p -> p.subtaskId));
            IterationState state = statsBuffer.get(0).state;
            IterationState nextState = state.copy();
            boolean shouldStop;

            if (state.phase == PHASE_SAMPLE) {
                nextState.candidateSets = solveLocalSamples();
                nextState.targetK = targetK;
                nextState.phase = PHASE_EVALUATE;
                shouldStop = false;
            } else {
                double[] aggregatedCosts = new double[state.candidateSets.length / state.targetK];
                for (PartialStats stats : statsBuffer) {
                    for (int set = 0; set < aggregatedCosts.length; set++) {
                        aggregatedCosts[set] += stats.candidateCosts[set];
                    }
                }

                int bestSetIdx = 0;
                for (int set = 1; set < aggregatedCosts.length; set++) {
                    if (aggregatedCosts[set] < aggregatedCosts[bestSetIdx]) {
                        bestSetIdx = set;
                    }
                }

                nextState.bestMedoidSet = new double[state.targetK][];
                System.arraycopy(state.candidateSets, bestSetIdx * state.targetK, nextState.bestMedoidSet, 0, state.targetK);
                nextState.minCost = aggregatedCosts[bestSetIdx];
                nextState.candidateSets = null;
                shouldStop = true;
                logger.debug("clara: winner=sample {} of {} cost={}", bestSetIdx, aggregatedCosts.length, aggregatedCosts[bestSetIdx]);
            }
            statsBuffer = null;

            IterationUpdate update = new IterationUpdate();
            update.state = nextState;
            update.shouldStop = shouldStop;
            out.collect(update);
        }

        private double[][] solveLocalSamples() {
            List<List<double[]>> groupedCoords = new ArrayList<>(numSamples);
            List<List<Double>> groupedWeights = new ArrayList<>(numSamples);
            for (int s = 0; s < numSamples; s++) {
                groupedCoords.add(new ArrayList<>());
                groupedWeights.add(new ArrayList<>());
            }

            for (PartialStats stats : statsBuffer) {
                for (int i = 0; i < stats.sampleIndices.length; i++) {
                    groupedCoords.get(stats.sampleIndices[i]).add(stats.sampledCoords[i]);
                    groupedWeights.get(stats.sampleIndices[i]).add(stats.sampledWeights[i]);
                }
            }

            double[][] allCandidates = new double[numSamples * targetK][];
            for (int s = 0; s < numSamples; s++) {
                List<double[]> currentCoords = groupedCoords.get(s);
                if (currentCoords.size() < targetK) {
                    throw new IllegalStateException("Sample " + s + " too small: got " + currentCoords.size() +
                        " points, need at least k=" + targetK + ". Increase sampleSize.");
                }

                int[] shuffleOrder = DriverSample.generateRandomPermutation(currentCoords.size(), seed + s);
                int keepCount = Math.min(sampleSize, currentCoords.size());

                double[][] selectedFeatures = new double[keepCount][];
                double[] selectedWeights = new double[keepCount];

                for (int i = 0; i < keepCount; i++) {
                    int targetIdx = currentCoords.size() <= sampleSize ? i : shuffleOrder[i];
                    selectedFeatures[i] = currentCoords.get(targetIdx);
                    selectedWeights[i] = groupedWeights.get(s).get(targetIdx);
                }

                DenseVector[] localMedoids = localSolver.fitLocal(selectedFeatures, selectedWeights).medoids();
                for (int i = 0; i < targetK; i++) {
                    allCandidates[s * targetK + i] = localMedoids[i].values;
                }
            }
            return allCandidates;
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
        public int targetK;
        public double[][] candidateSets;
        public double[][] bestMedoidSet;
        public double minCost;

        public IterationState() {}

        static IterationState initialize() {
            IterationState state = new IterationState();
            state.phase = PHASE_SAMPLE;
            state.minCost = Double.MAX_VALUE;
            return state;
        }

        IterationState copy() {
            IterationState state = new IterationState();
            state.phase = phase;
            state.targetK = targetK;
            state.candidateSets = candidateSets;
            state.bestMedoidSet = bestMedoidSet;
            state.minCost = minCost;
            return state;
        }
    }

    public static final class PartialStats implements Serializable {
        public int subtaskId;
        public int[] sampleIndices;
        public double[][] sampledCoords;
        public double[] sampledWeights;
        public double[] candidateCosts;
        public IterationState state;

        public PartialStats() {}
    }

    public static final class IterationUpdate implements Serializable {
        public IterationState state;
        public boolean shouldStop;

        public IterationUpdate() {}
    }
}
