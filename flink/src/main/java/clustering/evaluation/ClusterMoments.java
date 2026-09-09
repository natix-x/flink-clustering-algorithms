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

/** Per-cluster first and second moments about the cluster's own centroid — everything the
 *  centroid-based internal indices need, computed once and shared.
 *
 *  {@link DaviesBouldinIndex} wants the mean distance to the centroid, {@link CalinskiHarabaszIndex}
 *  the sum of squared distances, and both want the centroids and the cluster masses, so the two
 *  passes below run once for the pair rather than once per index
 *  ({@link clustering.benchmark.evaluation.EvaluationRunner} does the sharing). Byte-for-byte the
 *  same quantities the Spark {@code ClusterMoments} produces, so the two engines' {@code
 *  daviesBouldin} / {@code calinskiHarabasz} columns are comparable.
 *
 *  Cost is O(n·d) — two linear passes, no pairwise term — which is why, unlike the silhouette,
 *  these indices are computed on the FULL dataset and ignore {@code sampleSize}.
 *
 *  <h3>Two jobs, and what that costs on Flink</h3>
 *  Pass 2 needs the centroids pass 1 produces, so this is two Flink jobs and therefore two READS
 *  of the source — Flink has no cross-job cache, the same asymmetry {@code FlinkClusteringJob}
 *  documents for every multi-job algorithm (Spark reads once behind the caller's persist). The
 *  in-pattern alternative that would read once is a FLIP-176 job whose two ROUNDS are the two
 *  passes, with the points in a {@link org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache}
 *  — exactly what {@code EpsilonNeighbourCounter} does for the ε-scan's chunks. It is not done here
 *  yet on purpose: per hard rule 3 that shape needs a measurement showing the plain P1 version is
 *  the bottleneck, and the eval phase already re-reads the source for the label stats and the
 *  silhouette draw, so moving one of four reads is not where the phase's time goes. Revisit when
 *  {@code phaseDurationsMs} carries {@code eval.*} sub-keys.
 *
 *  <h3>Which centre is scored</h3>
 *  The centroid is the arithmetic WEIGHTED MEAN whatever {@code distance} says — scikit-learn's
 *  convention, which keeps the indices comparable with the reference implementation, and identical
 *  to the Spark side. It is NOT always the centre the algorithm optimised: under
 *  {@code geometry: spherical} the model's prototype is the normalised mean of normalised vectors,
 *  and the whole k-medoids ladder carries a MEDOID (under {@code manhattan} the dispersion-minimising
 *  centre is the componentwise median). In those runs DB and CH score a centre the run never
 *  produced, which inflates the dispersion terms. Deliberate, and noted the same way in the Spark
 *  implementation. */
public final class ClusterMoments {

    /** Label of a noise point; excluded from every index, exactly as it is from the silhouette. */
    static final int NOISE_LABEL = -1;

    /** One entry per non-noise cluster, ascending by label. */
    public final ClusterMoment[] clusterMoments;

    /** Weighted mean of the non-noise points — the grand centroid CH needs. Empty when the
     *  labelling produced no non-noise cluster at all. */
    public final double[] globalCentroid;

    public ClusterMoments(ClusterMoment[] clusterMoments, double[] globalCentroid) {
        this.clusterMoments = clusterMoments;
        this.globalCentroid = globalCentroid;
    }

    /** Number of non-noise clusters actually present in the labelling — NOT the configured k,
     *  which an algorithm may undershoot by emptying a cluster. */
    public int numClusters() {
        return clusterMoments.length;
    }

    /** Total mass of the non-noise points; equals their COUNT on unweighted input. */
    public double totalWeight() {
        double total = 0.0;
        for (ClusterMoment moment : clusterMoments) {
            total += moment.clusterWeight;
        }
        return total;
    }

    /** Moments of one cluster. Sums are weighted, so on unweighted input (every weight 1.0) they
     *  are the plain textbook sums, and on weighted input a point of weight w counts exactly as w
     *  copies of itself — the repo-wide weighting == duplication invariant. */
    public static final class ClusterMoment {

        public final int label;
        /** Weighted mean of the cluster's points. */
        public final double[] centroid;
        /** Σ w — the cluster's mass (its size, unweighted). */
        public final double clusterWeight;
        /** Σ w·d(x, centroid). */
        public final double sumOfWeightedDistances;
        /** Σ w·d(x, centroid)². */
        public final double sumOfWeightedSquaredDistances;

        public ClusterMoment(int label, double[] centroid, double clusterWeight,
                             double sumOfWeightedDistances, double sumOfWeightedSquaredDistances) {
            this.label = label;
            this.centroid = centroid;
            this.clusterWeight = clusterWeight;
            this.sumOfWeightedDistances = sumOfWeightedDistances;
            this.sumOfWeightedSquaredDistances = sumOfWeightedSquaredDistances;
        }

        /** Dispersion S_i: the mean distance of the cluster's points to its centroid. */
        public double meanDistance() {
            return clusterWeight <= 0.0 ? 0.0 : sumOfWeightedDistances / clusterWeight;
        }
    }

    /** Two passes over the labelled data: centroids and masses first, then the distance sums
     *  against those centroids. */
    public static ClusterMoments compute(Model model, PointSource source, EnvFactory envs,
                                         DistanceMetric distance) {
        // Pass 1 — centroid and mass per cluster.
        List<CentroidSum> centroidSums = new ArrayList<>();
        StreamExecutionEnvironment centroidEnv = envs.newEnv();
        DataStream<CentroidSum> sums = source.create(centroidEnv)
            .flatMap(new CentroidSumMap(model)).returns(CentroidSum.class)
            .keyBy(sum -> sum.label)
            .reduce((a, b) -> CentroidSum.plus(a, b));
        centroidSums.addAll(FlinkJobs.collectAll(sums, "cluster-moments-centroids"));

        if (centroidSums.isEmpty()) {
            return new ClusterMoments(new ClusterMoment[0], new double[0]);
        }
        // Ascending label, so the driver-side layout (and with it the index arithmetic below and
        // the labels the indices report) does not depend on which subtask finished first.
        centroidSums.sort(Comparator.comparingInt(sum -> sum.label));

        Map<Integer, double[]> centroids = new HashMap<>();
        for (CentroidSum sum : centroidSums) {
            centroids.put(sum.label, sum.centroid());
        }

        // Pass 2 — Σ w·d and Σ w·d² per cluster, against those centroids. The centroids travel in
        // the function's closure (k·d doubles), the counterpart of the Spark side's broadcast.
        StreamExecutionEnvironment distanceEnv = envs.newEnv();
        DataStream<DistanceSum> distanceSums = source.create(distanceEnv)
            .flatMap(new DistanceSumMap(model, centroids, distance)).returns(DistanceSum.class)
            .keyBy(sum -> sum.label)
            .reduce((a, b) -> DistanceSum.plus(a, b));
        Map<Integer, DistanceSum> byLabel = new HashMap<>();
        for (DistanceSum sum : FlinkJobs.collectAll(distanceSums, "cluster-moments-distances")) {
            byLabel.put(sum.label, sum);
        }

        ClusterMoment[] moments = new ClusterMoment[centroidSums.size()];
        for (int i = 0; i < moments.length; i++) {
            CentroidSum sum = centroidSums.get(i);
            // A cluster pass 1 found but pass 2 did not is the dangerous half of an unstable
            // labelling: defaulting to (0, 0) would report that cluster as having ZERO dispersion,
            // which deflates DB and inflates CH with nothing in the log to show for it.
            DistanceSum distances = byLabel.get(sum.label);
            if (distances == null) {
                throw new IllegalStateException("cluster " + sum.label
                    + " vanished between the two moment passes: the labelling is not stable across scans");
            }
            moments[i] = new ClusterMoment(sum.label, sum.centroid(), sum.weight,
                distances.sumOfWeightedDistances, distances.sumOfWeightedSquaredDistances);
        }
        return new ClusterMoments(moments, globalCentroidOf(moments));
    }

    /** Weighted mean of the cluster centroids — exactly the weighted mean of the underlying
     *  points, since each centroid is already its cluster's weighted mean, so no extra pass. */
    private static double[] globalCentroidOf(ClusterMoment[] moments) {
        int dimensions = moments[0].centroid.length;
        double[] coordinateSums = new double[dimensions];
        double totalWeight = 0.0;
        for (ClusterMoment moment : moments) {
            for (int d = 0; d < dimensions; d++) {
                coordinateSums[d] += moment.clusterWeight * moment.centroid[d];
            }
            totalWeight += moment.clusterWeight;
        }
        if (totalWeight > 0.0) {
            for (int d = 0; d < dimensions; d++) {
                coordinateSums[d] /= totalWeight;
            }
        }
        return coordinateSums;
    }

    /** Labels a point and emits its weighted coordinates; noise is dropped here, so the whole
     *  index is computed on the clustered points only (identically on both engines). */
    static final class CentroidSumMap implements FlatMapFunction<WeightedPoint, CentroidSum> {
        private final Model model;

        CentroidSumMap(Model model) {
            this.model = model;
        }

        @Override
        public void flatMap(WeightedPoint point, Collector<CentroidSum> out) {
            int label = model.predict(point.features);
            if (label == NOISE_LABEL) {
                return;
            }
            // `values` is the vector's own array — read once, outside any loop.
            double[] x = point.features.values;
            double w = point.weight;
            double[] weighted = new double[x.length];
            for (int d = 0; d < x.length; d++) {
                weighted[d] = w * x[d];
            }
            CentroidSum sum = new CentroidSum();
            sum.label = label;
            sum.weight = w;
            sum.weightedCoordinateSum = weighted;
            out.collect(sum);
        }
    }

    /** Second pass: the point's distance to its own cluster's centroid, weighted. */
    static final class DistanceSumMap implements FlatMapFunction<WeightedPoint, DistanceSum> {
        private final Model model;
        private final Map<Integer, double[]> centroids;
        private final DistanceMetric distance;

        DistanceSumMap(Model model, Map<Integer, double[]> centroids, DistanceMetric distance) {
            this.model = model;
            this.centroids = centroids;
            this.distance = distance;
        }

        @Override
        public void flatMap(WeightedPoint point, Collector<DistanceSum> out) {
            int label = model.predict(point.features);
            if (label == NOISE_LABEL) {
                return;
            }
            double[] centroid = centroids.get(label);
            if (centroid == null) {
                throw new IllegalStateException("pass 2 saw cluster " + label
                    + ", which pass 1 did not produce: the labelling is not stable across scans, so "
                    + "these moments would be computed against the wrong centroids");
            }
            double d = distance.compute(point.features.values, centroid);
            DistanceSum sum = new DistanceSum();
            sum.label = label;
            sum.sumOfWeightedDistances = point.weight * d;
            sum.sumOfWeightedSquaredDistances = point.weight * d * d;
            out.collect(sum);
        }
    }

    /** Per-label (Σ w, Σ w·x), shuffled through the pass-1 keyBy/reduce.
     *  Public static so Flink's TypeExtractor treats it as a POJO (fast path, not Kryo). */
    public static final class CentroidSum {
        public int label;
        public double weight;
        public double[] weightedCoordinateSum;

        public CentroidSum() {}

        static CentroidSum plus(CentroidSum a, CentroidSum b) {
            CentroidSum out = new CentroidSum();
            out.label = a.label;
            out.weight = a.weight + b.weight;
            out.weightedCoordinateSum = new double[a.weightedCoordinateSum.length];
            for (int d = 0; d < out.weightedCoordinateSum.length; d++) {
                out.weightedCoordinateSum[d] = a.weightedCoordinateSum[d] + b.weightedCoordinateSum[d];
            }
            return out;
        }

        double[] centroid() {
            double[] centroid = new double[weightedCoordinateSum.length];
            if (weight <= 0.0) {
                return centroid;
            }
            for (int d = 0; d < centroid.length; d++) {
                centroid[d] = weightedCoordinateSum[d] / weight;
            }
            return centroid;
        }
    }

    /** Per-label (Σ w·d, Σ w·d²), shuffled through the pass-2 keyBy/reduce. */
    public static final class DistanceSum {
        public int label;
        public double sumOfWeightedDistances;
        public double sumOfWeightedSquaredDistances;

        public DistanceSum() {}

        static DistanceSum plus(DistanceSum a, DistanceSum b) {
            DistanceSum out = new DistanceSum();
            out.label = a.label;
            out.sumOfWeightedDistances = a.sumOfWeightedDistances + b.sumOfWeightedDistances;
            out.sumOfWeightedSquaredDistances =
                a.sumOfWeightedSquaredDistances + b.sumOfWeightedSquaredDistances;
            return out;
        }
    }

}
