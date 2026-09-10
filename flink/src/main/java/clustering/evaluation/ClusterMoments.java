package clustering.evaluation;

import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.core.WeightedPoint;
import clustering.distance.DistanceMetric;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Computes per-cluster first and second moments about the cluster's centroid.
 * Required for internal evaluation indices (e.g., Davies-Bouldin, Calinski-Harabasz).
 *
 * Execution requires two Flink jobs (passes):
 * 1. Calculate cluster centroids and total weights.
 * 2. Calculate distance sums against the newly computed centroids.
 */
public final class ClusterMoments {

    static final int NOISE_LABEL = -1;

    public final ClusterMoment[] clusterMoments;
    public final double[] globalCentroid;

    public ClusterMoments(ClusterMoment[] clusterMoments, double[] globalCentroid) {
        this.clusterMoments = clusterMoments;
        this.globalCentroid = globalCentroid;
    }

    public int numClusters() {
        return clusterMoments.length;
    }

    public double totalWeight() {
        double total = 0.0;
        for (ClusterMoment moment : clusterMoments) {
            total += moment.clusterWeight;
        }
        return total;
    }

    public static final class ClusterMoment {
        public final int label;
        public final double[] centroid;
        public final double clusterWeight;
        public final double sumOfWeightedDistances;
        public final double sumOfWeightedSquaredDistances;

        public ClusterMoment(int label, double[] centroid, double clusterWeight,
                             double sumOfWeightedDistances, double sumOfWeightedSquaredDistances) {
            this.label = label;
            this.centroid = centroid;
            this.clusterWeight = clusterWeight;
            this.sumOfWeightedDistances = sumOfWeightedDistances;
            this.sumOfWeightedSquaredDistances = sumOfWeightedSquaredDistances;
        }

        public double meanDistance() {
            return clusterWeight <= 0.0 ? 0.0 : sumOfWeightedDistances / clusterWeight;
        }
    }

    public static ClusterMoments compute(Model model, PointSource source, EnvFactory envFactory,
                                         DistanceMetric distanceMetric) {

        // Pass 1: Compute centroids and mass per cluster
        StreamExecutionEnvironment centroidEnv = envFactory.newEnv();
        DataStream<CentroidSum> centroidSumsStream = source.create(centroidEnv)
            .flatMap(new MapToCentroidSum(model)).returns(CentroidSum.class)
            .keyBy(sum -> sum.label)
            .reduce(CentroidSum::plus);

        List<CentroidSum> centroidSums = new ArrayList<>(FlinkJobs.collectAll(centroidSumsStream, "cluster-moments-centroids"));

        if (centroidSums.isEmpty()) {
            return new ClusterMoments(new ClusterMoment[0], new double[0]);
        }

        // Sort ascending by label to ensure deterministic driver-side layout
        centroidSums.sort(Comparator.comparingInt(sum -> sum.label));

        Map<Integer, double[]> centroidByLabel = new HashMap<>();
        for (CentroidSum sum : centroidSums) {
            centroidByLabel.put(sum.label, sum.centroid());
        }

        // Pass 2: Compute weighted distance sums against pass 1 centroids
        StreamExecutionEnvironment distanceEnv = envFactory.newEnv();
        DataStream<DistanceSum> distanceSumsStream = source.create(distanceEnv)
            .flatMap(new MapToDistanceSum(model, centroidByLabel, distanceMetric)).returns(DistanceSum.class)
            .keyBy(sum -> sum.label)
            .reduce(DistanceSum::plus);

        Map<Integer, DistanceSum> distanceSumsByLabel = new HashMap<>();
        for (DistanceSum sum : FlinkJobs.collectAll(distanceSumsStream, "cluster-moments-distances")) {
            distanceSumsByLabel.put(sum.label, sum);
        }

        ClusterMoment[] computedMoments = new ClusterMoment[centroidSums.size()];
        for (int i = 0; i < computedMoments.length; i++) {
            CentroidSum centroidSum = centroidSums.get(i);
            DistanceSum distanceSum = distanceSumsByLabel.get(centroidSum.label);

            if (distanceSum == null) {
                throw new IllegalStateException("Cluster " + centroidSum.label
                    + " vanished between moment passes. The labelling is unstable across scans.");
            }

            computedMoments[i] = new ClusterMoment(
                centroidSum.label,
                centroidSum.centroid(),
                centroidSum.weight,
                distanceSum.sumOfWeightedDistances,
                distanceSum.sumOfWeightedSquaredDistances
            );
        }

        return new ClusterMoments(computedMoments, computeGlobalCentroid(computedMoments));
    }

    private static double[] computeGlobalCentroid(ClusterMoment[] moments) {
        int dimensions = moments[0].centroid.length;
        double[] coordinateSums = new double[dimensions];
        double totalWeight = 0.0;

        for (ClusterMoment moment : moments) {
            for (int dim = 0; dim < dimensions; dim++) {
                coordinateSums[dim] += moment.clusterWeight * moment.centroid[dim];
            }
            totalWeight += moment.clusterWeight;
        }

        if (totalWeight > 0.0) {
            for (int dim = 0; dim < dimensions; dim++) {
                coordinateSums[dim] /= totalWeight;
            }
        }
        return coordinateSums;
    }

    static final class MapToCentroidSum implements FlatMapFunction<WeightedPoint, CentroidSum> {
        private final Model model;

        MapToCentroidSum(Model model) {
            this.model = model;
        }

        @Override
        public void flatMap(WeightedPoint point, Collector<CentroidSum> out) {
            int label = model.predict(point.features);
            if (label == NOISE_LABEL) {
                return;
            }

            double[] coordinates = point.features.values;
            double pointWeight = point.weight;
            double[] weightedCoordinates = new double[coordinates.length];

            for (int dim = 0; dim < coordinates.length; dim++) {
                weightedCoordinates[dim] = pointWeight * coordinates[dim];
            }

            CentroidSum sum = new CentroidSum();
            sum.label = label;
            sum.weight = pointWeight;
            sum.weightedCoordinateSum = weightedCoordinates;
            out.collect(sum);
        }
    }

    static final class MapToDistanceSum implements FlatMapFunction<WeightedPoint, DistanceSum> {
        private final Model model;
        private final Map<Integer, double[]> centroids;
        private final DistanceMetric distanceMetric;

        MapToDistanceSum(Model model, Map<Integer, double[]> centroids, DistanceMetric distanceMetric) {
            this.model = model;
            this.centroids = centroids;
            this.distanceMetric = distanceMetric;
        }

        @Override
        public void flatMap(WeightedPoint point, Collector<DistanceSum> out) {
            int label = model.predict(point.features);
            if (label == NOISE_LABEL) {
                return;
            }

            double[] centroid = centroids.get(label);
            if (centroid == null) {
                throw new IllegalStateException("Pass 2 encountered cluster " + label
                    + " which Pass 1 did not produce. The labelling is unstable.");
            }

            double distanceToCentroid = distanceMetric.compute(point.features.values, centroid);

            DistanceSum sum = new DistanceSum();
            sum.label = label;
            sum.sumOfWeightedDistances = point.weight * distanceToCentroid;
            sum.sumOfWeightedSquaredDistances = point.weight * distanceToCentroid * distanceToCentroid;
            out.collect(sum);
        }
    }

    /**
     * Must remain a public POJO (public fields, no-arg constructor)
     * so Flink's TypeExtractor applies fast-path serialization instead of Kryo.
     */
    public static final class CentroidSum {
        public int label;
        public double weight;
        public double[] weightedCoordinateSum;

        public CentroidSum() {}

        static CentroidSum plus(CentroidSum sumA, CentroidSum sumB) {
            CentroidSum combinedSum = new CentroidSum();
            combinedSum.label = sumA.label;
            combinedSum.weight = sumA.weight + sumB.weight;
            combinedSum.weightedCoordinateSum = new double[sumA.weightedCoordinateSum.length];

            for (int dim = 0; dim < combinedSum.weightedCoordinateSum.length; dim++) {
                combinedSum.weightedCoordinateSum[dim] = sumA.weightedCoordinateSum[dim] + sumB.weightedCoordinateSum[dim];
            }
            return combinedSum;
        }

        double[] centroid() {
            double[] calculatedCentroid = new double[weightedCoordinateSum.length];
            if (weight <= 0.0) {
                return calculatedCentroid;
            }
            for (int dim = 0; dim < calculatedCentroid.length; dim++) {
                calculatedCentroid[dim] = weightedCoordinateSum[dim] / weight;
            }
            return calculatedCentroid;
        }
    }

    /**
     * Must remain a public POJO for Flink serialization.
     */
    public static final class DistanceSum {
        public int label;
        public double sumOfWeightedDistances;
        public double sumOfWeightedSquaredDistances;

        public DistanceSum() {}

        static DistanceSum plus(DistanceSum sumA, DistanceSum sumB) {
            DistanceSum combinedSum = new DistanceSum();
            combinedSum.label = sumA.label;
            combinedSum.sumOfWeightedDistances = sumA.sumOfWeightedDistances + sumB.sumOfWeightedDistances;
            combinedSum.sumOfWeightedSquaredDistances = sumA.sumOfWeightedSquaredDistances + sumB.sumOfWeightedSquaredDistances;
            return combinedSum;
        }
    }
}
