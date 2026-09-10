package clustering.evaluation;

import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.core.WeightedPoint;
import clustering.core.Weights;
import clustering.distance.DistanceMetric;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.stream.IntStream;

/**
 * Computes the mass-weighted mean silhouette score across clustered data points.
 * Evaluates distances using a subsampled representative set to bound the O(n²) complexity,
 * choosing between a driver-local multi-core scan and a distributed Flink job.
 */
public final class SilhouetteEvaluator implements ClusteringEvaluator {

    private static final Logger logger = LoggerFactory.getLogger(SilhouetteEvaluator.class);

    private static final int NOISE_LABEL = -1;
    private static final double COMPARISONS_PER_CORE_SECOND = 1.0e9;
    private static final double MIN_DRIVER_SECONDS_FOR_CLUSTER = 30.0;
    private static final int MIN_PARALLELISM_RATIO = 2;
    private static final long MAX_SHIPPED_COORDS_BYTES = 64L * 1024 * 1024;
    private static final long MAX_PARTIAL_BUFFER_BYTES = 256L * 1024 * 1024;

    private final DistanceMetric distanceMetric;

    public SilhouetteEvaluator(DistanceMetric distanceMetric) {
        this.distanceMetric = distanceMetric;
    }

    @Override
    public double evaluate(Model model, PointSource source, EnvFactory envFactory) {
        return measure(model, source, envFactory, 1, null, 0L, null).score;
    }

    /**
     * Computes the silhouette evaluation outcome on a sampled dataset.
     */
    public Outcome measure(Model model, PointSource source, EnvFactory envFactory, int parallelism,
                           Integer sampleSize, long seed, Population population) {

        List<Drawn> drawnPoints = drawSample(model, source, envFactory, sampleSize, seed, population);
        Collected collected = collectFrom(drawnPoints, population == null ? null : population.clusterMasses);
        Sample sample = collected.sample;

        if (sample.getNumClusters() < 2) {
            return new Outcome(0.0, 0, sample.getNumClusters(), sample.size() + collected.droppedRows);
        }

        double[] perPointScores = computeAllScores(sample, envFactory, parallelism);

        double totalScoreWeight = 0.0;
        double totalMass = 0.0;
        int scoredCount = 0;
        int unscoredCount = 0;

        for (int i = 0; i < perPointScores.length; i++) {
            if (Double.isNaN(perPointScores[i])) {
                unscoredCount++;
            } else {
                double weight = sample.weights[i];
                totalScoreWeight += weight * perPointScores[i];
                totalMass += weight;
                scoredCount++;
            }
        }

        double score = totalMass <= 0.0 ? 0.0 : totalScoreWeight / totalMass;
        return new Outcome(
            score,
            scoredCount,
            sample.getNumClusters(),
            unscoredCount + collected.droppedRows);
    }

    private List<Drawn> drawSample(Model model, PointSource source, EnvFactory envFactory,
                                       Integer sampleSize, long seed, Population population) {
        boolean shouldSample = sampleSize != null && sampleSize > 0;
        double perMassSamplingRate = Double.NaN;

        if (shouldSample) {
            double totalMass = population != null ? population.totalMass : computeLabelledMass(model, source, envFactory);
            if (totalMass > sampleSize) {
                perMassSamplingRate = sampleSize / totalMass;
            }
        }

        StreamExecutionEnvironment env = envFactory.newEnv();
        DataStream<Drawn> drawnStream = source.create(env)
            .flatMap(new DrawMap(model, perMassSamplingRate, seed, shouldSample ? sampleSize : Integer.MAX_VALUE))
            .returns(Drawn.class)
            .name("silhouette-draw");

        return FlinkJobs.collectAll(drawnStream, "silhouette-draw");
    }

    private double computeLabelledMass(Model model, PointSource source, EnvFactory envFactory) {
        StreamExecutionEnvironment env = envFactory.newEnv();
        DataStream<Double> massStream = source.create(env)
            .flatMap((FlatMapFunction<WeightedPoint, Double>) (point, out) -> {
                if (model.predict(point.features) != NOISE_LABEL) {
                    double weight = Weights.sanitize(point.weight);
                    if (weight > 0.0) {
                        out.collect(weight);
                    }
                }
            })
            .returns(Types.DOUBLE)
            .keyBy(x -> 0)
            .reduce(Double::sum);

        Double aggregatedMass = FlinkJobs.last(massStream, "silhouette-mass");
        return aggregatedMass == null ? 0.0 : aggregatedMass;
    }

    static double rowUniform(double[] coordinates, double weight, long seed) {
        long hash = finalizeSplitMix64(seed + 0x9E3779B97F4A7C15L);
        for (double coord : coordinates) {
            hash = finalizeSplitMix64(hash ^ Double.doubleToLongBits(coord));
        }
        hash = finalizeSplitMix64(hash ^ Double.doubleToLongBits(weight));
        return (hash >>> 11) * 0x1.0p-53;
    }

    private static long finalizeSplitMix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final class DrawMap implements FlatMapFunction<WeightedPoint, Drawn> {

        private final Model model;
        private final double perMassSamplingRate;
        private final long seed;
        private final int copyLimit;

        DrawMap(Model model, double perMassSamplingRate, long seed, int copyLimit) {
            this.model = model;
            this.perMassSamplingRate = perMassSamplingRate;
            this.seed = seed;
            this.copyLimit = copyLimit;
        }

        @Override
        public void flatMap(WeightedPoint point, Collector<Drawn> out) {
            int clusterLabel = model.predict(point.features);
            if (clusterLabel == NOISE_LABEL) {
                return;
            }

            double weight = Weights.sanitize(point.weight);
            if (weight <= 0.0) {
                return;
            }

            double[] coords = point.features.values;

            if (Double.isNaN(perMassSamplingRate)) {
                out.collect(new Drawn(clusterLabel, coords.clone(), weight));
                return;
            }

            double expectedCopies = weight * perMassSamplingRate;
            double wholeCopies = Math.floor(expectedCopies);
            long numCopies = (long) wholeCopies
                + (rowUniform(coords, weight, seed) < expectedCopies - wholeCopies ? 1L : 0L);

            numCopies = Math.max(0L, Math.min(numCopies, copyLimit));
            for (long c = 0; c < numCopies; c++) {
                out.collect(new Drawn(clusterLabel, coords.clone(), 1.0));
            }
        }
    }

    private double[] computeAllScores(Sample sample, EnvFactory envFactory, int parallelism) {
        int sampleSize = sample.size();
        int dimension = sampleSize == 0 ? 0 : sample.points[0].length;
        long payloadBytes = (long) sampleSize * dimension * 8L;
        int driverCores = Runtime.getRuntime().availableProcessors();
        String localReason = planFor(sampleSize, dimension, parallelism, driverCores, payloadBytes);

        if (localReason == null) {
            logger.info("silhouette: scoring {} sample points across the cluster ({} coords, {} MB payload)",
                sampleSize, dimension, payloadBytes / 1e6);
            return scoreDistributed(sample, envFactory, parallelism);
        }

        logger.info("silhouette: scoring {} sample points on the driver — {}", sampleSize, localReason);
        return scoreLocally(sample);
    }

    static String planFor(int sampleSize, int dimension, int clusterParallelism,
                                         int driverCores, long payloadBytes) {
        double totalPairs = (double) sampleSize * sampleSize;
        double estimatedDriverSeconds =
            totalPairs * Math.max(1, dimension) / (Math.max(1, driverCores) * COMPARISONS_PER_CORE_SECOND);

        if (estimatedDriverSeconds < MIN_DRIVER_SECONDS_FOR_CLUSTER) {
            return String.format(
                "%d points score in ~%.0f s on %d driver cores (below threshold %.0f s)",
                sampleSize, estimatedDriverSeconds, driverCores, MIN_DRIVER_SECONDS_FOR_CLUSTER);
        }
        if (clusterParallelism < MIN_PARALLELISM_RATIO * Math.max(1, driverCores)) {
            return String.format(
                "Cluster parallelism %d does not justify overhead against %d driver cores",
                clusterParallelism, driverCores);
        }
        if (payloadBytes > MAX_SHIPPED_COORDS_BYTES) {
            return String.format(
                "Payload size %.1f GB exceeds maximum shipping %.1f GB",
                payloadBytes / 1e9, MAX_SHIPPED_COORDS_BYTES / 1e9);
        }
        return null;
    }

    double[] scoreLocally(Sample sample) {
        int n = sample.size();
        if (n == 0) {
            return new double[0];
        }
        int requestedWorkers = Math.max(1, Math.min(Runtime.getRuntime().availableProcessors(), n));
        double[][] distanceSums = sample.computeSymmetricDistanceSums(distanceMetric, requestedWorkers);
        return IntStream.range(0, n).parallel()
            .mapToDouble(i -> sample.finalizeScore(i, distanceSums[i]))
            .toArray();
    }

    double[] scoreDistributed(Sample sample, EnvFactory envFactory, int parallelism) {
        StreamExecutionEnvironment env = envFactory.newEnv();
        env.setParallelism(Math.max(1, parallelism));

        DataStream<EvaluatedPointScore> scoreStream = env
            .fromSequence(0L, sample.size() - 1L)
            .map(new ScoreIndexMapper(sample, distanceMetric))
            .returns(EvaluatedPointScore.class)
            .name("silhouette-score");

        double[] results = new double[sample.size()];
        Arrays.fill(results, Double.NaN);
        FlinkJobs.consume(scoreStream, "silhouette-score", record -> results[record.pointIndex] = record.calculatedScore);
        return results;
    }

    static final class ScoreIndexMapper implements MapFunction<Long, EvaluatedPointScore> {

        private final Sample sample;
        private final DistanceMetric distanceMetric;
        private transient double[] scratchBuffer;

        ScoreIndexMapper(Sample sample, DistanceMetric distanceMetric) {
            this.sample = sample;
            this.distanceMetric = distanceMetric;
        }

        @Override
        public EvaluatedPointScore map(Long sequenceIndex) {
            if (scratchBuffer == null) {
                scratchBuffer = new double[sample.getNumClusters()];
            }
            int index = sequenceIndex.intValue();
            return new EvaluatedPointScore(index, sample.score(index, scratchBuffer, distanceMetric));
        }
    }

    static Collected collectFrom(List<Drawn> rawPoints, Map<Integer, Double> knownClusterMasses) {
        List<Drawn> validPoints = new ArrayList<>(rawPoints.size());
        int droppedRows = 0;

        for (Drawn point : rawPoints) {
            if (Double.isFinite(point.weight) && point.weight > 0.0 && areAllCoordinatesFinite(point.coordinates)) {
                validPoints.add(point);
            } else {
                droppedRows++;
            }
        }
        validPoints.sort(DRAWN_POINT_COMPARATOR);

        int totalSize = validPoints.size();
        double[][] points = new double[totalSize][];
        double[] weights = new double[totalSize];
        int[] labels = new int[totalSize];
        TreeSet<Integer> uniqueLabels = new TreeSet<>();

        for (int i = 0; i < totalSize; i++) {
            Drawn row = validPoints.get(i);
            points[i] = row.coordinates;
            weights[i] = row.weight;
            labels[i] = row.clusterLabel;
            uniqueLabels.add(row.clusterLabel);
        }

        Map<Integer, Integer> labelToSlotMap = new HashMap<>();
        for (Integer label : uniqueLabels) {
            labelToSlotMap.put(label, labelToSlotMap.size());
        }

        int[] assignedSlots = new int[totalSize];
        double[] sampleClusterMasses = new double[labelToSlotMap.size()];
        for (int i = 0; i < totalSize; i++) {
            assignedSlots[i] = labelToSlotMap.get(labels[i]);
            sampleClusterMasses[assignedSlots[i]] += weights[i];
        }

        double[] fullClusterMasses = new double[labelToSlotMap.size()];
        for (Map.Entry<Integer, Integer> entry : labelToSlotMap.entrySet()) {
            Double mass = knownClusterMasses == null ? null : knownClusterMasses.get(entry.getKey());
            fullClusterMasses[entry.getValue()] = mass == null ? sampleClusterMasses[entry.getValue()] : mass;
        }

        return new Collected(
            new Sample(points, weights, assignedSlots, sampleClusterMasses, fullClusterMasses),
            droppedRows);
    }

    private static final Comparator<Drawn> DRAWN_POINT_COMPARATOR = (a, b) -> {
        int labelComparison = Integer.compare(a.clusterLabel, b.clusterLabel);
        if (labelComparison != 0) {
            return labelComparison;
        }
        int sharedDims = Math.min(a.coordinates.length, b.coordinates.length);
        for (int d = 0; d < sharedDims; d++) {
            int coordComparison = Double.compare(a.coordinates[d], b.coordinates[d]);
            if (coordComparison != 0) {
                return coordComparison;
            }
        }
        int lengthComparison = Integer.compare(a.coordinates.length, b.coordinates.length);
        return lengthComparison != 0 ? lengthComparison : Double.compare(a.weight, b.weight);
    };

    private static boolean areAllCoordinatesFinite(double[] coords) {
        for (double val : coords) {
            if (!Double.isFinite(val)) {
                return false;
            }
        }
        return true;
    }

    public static final class Population implements Serializable {

        public final Map<Integer, Double> clusterMasses;
        public final double totalMass;

        public Population(Map<Integer, Double> clusterMasses) {
            this.clusterMasses = clusterMasses;
            double massAccumulator = 0.0;
            for (double mass : clusterMasses.values()) {
                massAccumulator += mass;
            }
            this.totalMass = massAccumulator;
        }
    }

    public static final class Outcome {

        public final double score;
        public final int scoredPoints;
        public final int sampleClusters;
        public final int unscoredPoints;

        public Outcome(double score, int scoredPoints, int sampleClusters, int unscoredPoints) {
            this.score = score;
            this.scoredPoints = scoredPoints;
            this.sampleClusters = sampleClusters;
            this.unscoredPoints = unscoredPoints;
        }
    }

    public static final class Drawn {

        public int clusterLabel;
        public double[] coordinates;
        public double weight;

        public Drawn() {}

        public Drawn(int clusterLabel, double[] coordinates, double weight) {
            this.clusterLabel = clusterLabel;
            this.coordinates = coordinates;
            this.weight = weight;
        }
    }

    public static final class EvaluatedPointScore {

        public int pointIndex;
        public double calculatedScore;

        public EvaluatedPointScore() {}

        public EvaluatedPointScore(int pointIndex, double calculatedScore) {
            this.pointIndex = pointIndex;
            this.calculatedScore = calculatedScore;
        }
    }

    static final class Collected {

        final Sample sample;
        final int droppedRows;

        Collected(Sample sample, int droppedRows) {
            this.sample = sample;
            this.droppedRows = droppedRows;
        }
    }

    static final class Sample implements Serializable {

        final double[][] points;
        final double[] weights;
        final int[] slots;
        final double[] sampleClusterMasses;
        final double[] trueClusterMasses;

        Sample(double[][] points, double[] weights, int[] slots,
                  double[] sampleClusterMasses, double[] trueClusterMasses) {
            this.points = points;
            this.weights = weights;
            this.slots = slots;
            this.sampleClusterMasses = sampleClusterMasses;
            this.trueClusterMasses = trueClusterMasses;
        }

        int size() {
            return points.length;
        }

        int getNumClusters() {
            return sampleClusterMasses.length;
        }

        double score(int pointIdx, double[] distanceSums, DistanceMetric distanceMetric) {
            Arrays.fill(distanceSums, 0.0);

            double[] currentCoords = points[pointIdx];
            for (int j = 0; j < points.length; j++) {
                if (j == pointIdx) {
                    // d(x, x) = 0 for every metric in use here (Euclidean/Manhattan/Cosine/UnitSphere
                    // are all proper metrics) — skip the call instead of computing and subtracting it.
                    continue;
                }
                distanceSums[slots[j]] += weights[j] * distanceMetric.compute(currentCoords, points[j]);
            }

            return finalizeScore(pointIdx, distanceSums);
        }

        /**
         * Builds the full n x k per-cluster distance-sum matrix by visiting each unordered pair
         * (i, j) once: every metric in use is symmetric, so a single {@code compute(i, j)} feeds
         * both point i's and point j's running sum, halving the distance calls the naive
         * per-point scan makes.
         *
         * Rows are assigned to workers round-robin (i % workers) rather than in contiguous blocks
         * so the triangular workload (row i only ever scans j > i) balances evenly across workers.
         * Each worker accumulates into its OWN private n x k buffer — never into another worker's —
         * so there is no cross-thread write and no locking; the buffers are summed elementwise once
         * all workers finish. Worker count is capped so the private buffers together stay under a
         * fixed memory budget, since that cost scales with workers x n x k.
         */
        double[][] computeSymmetricDistanceSums(DistanceMetric distanceMetric, int requestedWorkers) {
            int n = size();
            int numClusters = getNumClusters();

            long bytesPerWorker = (long) n * numClusters * Double.BYTES;
            int maxWorkersByMemory = (int) Math.max(1, MAX_PARTIAL_BUFFER_BYTES / Math.max(1L, bytesPerWorker));
            int workers = Math.max(1, Math.min(Math.min(requestedWorkers, maxWorkersByMemory), n));

            double[][][] partials = new double[workers][][];
            IntStream.range(0, workers).parallel().forEach(w -> {
                double[][] local = new double[n][numClusters];
                for (int i = w; i < n; i += workers) {
                    double[] ci = points[i];
                    int slotI = slots[i];
                    double wi = weights[i];
                    for (int j = i + 1; j < n; j++) {
                        double d = distanceMetric.compute(ci, points[j]);
                        local[i][slots[j]] += weights[j] * d;
                        local[j][slotI] += wi * d;
                    }
                }
                partials[w] = local;
            });

            double[][] sums = new double[n][numClusters];
            for (double[][] partial : partials) {
                for (int i = 0; i < n; i++) {
                    double[] row = partial[i];
                    double[] target = sums[i];
                    for (int c = 0; c < numClusters; c++) {
                        target[c] += row[c];
                    }
                }
            }
            return sums;
        }

        double finalizeScore(int pointIdx, double[] distanceSums) {
            int ownSlot = slots[pointIdx];

            if (trueClusterMasses[ownSlot] <= 1.0) {
                return 0.0;
            }

            double selfContribution = Math.min(1.0, weights[pointIdx]);
            double intraDenominator = sampleClusterMasses[ownSlot] - selfContribution;
            if (intraDenominator <= 0.0) {
                return Double.NaN;
            }

            double intraMeanDistance = distanceSums[ownSlot] / intraDenominator;

            double minInterMeanDistance = Double.POSITIVE_INFINITY;
            for (int slot = 0; slot < distanceSums.length; slot++) {
                if (slot != ownSlot && sampleClusterMasses[slot] > 0.0) {
                    double interMean = distanceSums[slot] / sampleClusterMasses[slot];
                    if (Double.isFinite(interMean) && interMean < minInterMeanDistance) {
                        minInterMeanDistance = interMean;
                    }
                }
            }

            double scaleFactor = Math.max(intraMeanDistance, minInterMeanDistance);
            if (!Double.isFinite(scaleFactor) || scaleFactor <= 0.0) {
                return 0.0;
            }

            double score = (minInterMeanDistance - intraMeanDistance) / scaleFactor;
            return Double.isFinite(score) ? score : 0.0;
        }
    }
}
