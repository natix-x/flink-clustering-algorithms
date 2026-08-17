package clustering.algorithms.dbscan;

import clustering.core.Clusterer;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/** DBSCAN++ (Jang &amp; Jiang, <i>Sub-sampled DBSCAN</i>, ICML 2019). Java mirror of the Spark
 *  {@code clustering.algorithms.dbscan.DBSCANpp}.
 *
 *  Shrinks the GRAPH, not the data: only m = ceil(s*n) sampled candidates can become core
 *  points, so the component structure fits on the driver. Cost O(n*m) instead of O(n^2).
 *
 *  Four steps:
 *  <ol>
 *    <li>sample m candidates ({@link CandidateSelectionStrategy}, one Flink job);</li>
 *    <li>count each candidate's eps-neighbours against the FULL dataset
 *        ({@link EpsilonNeighbourCounter}, ONE Flink job with the points cached across chunk
 *        rounds) — exact densities; core point iff count &gt;= {@code minPts};</li>
 *    <li>connected components of the eps-graph over core points ({@link EpsilonGraphComponents}),
 *        O(m^2*d) — the m²/2 edge scan runs on the cluster once m makes a job worth launching,
 *        the union-find itself always on the driver;</li>
 *    <li>label remaining points by nearest core point ({@link CoreLabelModel}).</li>
 *  </ol>
 *
 *  With {@code coreSampleFraction = 1.0} and {@code requireWithinEps = true} the core set,
 *  cluster count and noise set are exactly classic DBSCAN (the {@code dbscanexact} registry
 *  entry). Border points within eps of two components go to the NEAREST core (DBSCAN itself
 *  picks by scan order) — deterministic by choice.
 *
 *  Cluster ids come from the smallest candidate index in each component, so they are
 *  reproducible rather than dependent on union-find root order.
 *
 *  @param coreSampleFraction s in (0, 1]; the accuracy-vs-cost knob (m = ceil(s*n))
 *  @param requireWithinEps   {@code true} = classic DBSCAN noise semantics, {@code false} = the
 *                            paper's no-noise assignment (see {@link CoreLabelModel})
 *  @param chunkSize          candidate coordinates resident per subtask during counting */
public class DBSCANpp implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(DBSCANpp.class);

    private final double eps;
    private final int minPts;
    private final double coreSampleFraction;
    private final CandidateSelectionStrategy samplingStrategy;
    private final boolean requireWithinEps;
    private final int chunkSize;
    private final DistanceMetric distanceMetric;
    private final long seed;

    public DBSCANpp(double eps, int minPts, double coreSampleFraction,
                    CandidateSelectionStrategy samplingStrategy, boolean requireWithinEps,
                    int chunkSize, DistanceMetric distanceMetric, long seed) {
        if (!(eps > 0.0) || Double.isNaN(eps)) {
            throw new IllegalArgumentException("eps must be > 0, got " + eps);
        }
        if (minPts < 1) {
            throw new IllegalArgumentException("minPts must be >= 1, got " + minPts);
        }
        if (!(coreSampleFraction > 0.0) || coreSampleFraction > 1.0) {
            throw new IllegalArgumentException(
                "coreSampleFraction must be in (0, 1], got " + coreSampleFraction);
        }
        if (chunkSize < 1) {
            throw new IllegalArgumentException("chunkSize must be >= 1, got " + chunkSize);
        }
        this.eps = eps;
        this.minPts = minPts;
        this.coreSampleFraction = coreSampleFraction;
        this.samplingStrategy = samplingStrategy;
        this.requireWithinEps = requireWithinEps;
        this.chunkSize = chunkSize;
        this.distanceMetric = distanceMetric;
        this.seed = seed;
    }

    @Override
    public CoreLabelModel fit(PointSource source, EnvFactory envs, int parallelism) {
        long datasetSize = Datasets.count(source, envs);

        // m is an int — every downstream structure (candidates, eps-graph, union-find) is
        // indexed on the driver, even where the scan over it is distributed.
        int candidateCount = (int) Math.max(1L,
            Math.min(datasetSize, (long) Math.ceil(coreSampleFraction * datasetSize)));

        // Step 1: candidate core points (P2).
        double[][] candidatePoints = samplingStrategy.selectCandidates(
            source, envs, datasetSize, candidateCount, seed, distanceMetric);
        logger.info("dbscanpp: datasetSize={} m={} sampling={} eps={} minPts={} s={}",
            datasetSize, candidatePoints.length, samplingStrategy.strategyName(),
            eps, minPts, coreSampleFraction);

        // Step 2: exact eps-neighbour counts against the full dataset (P1).
        double[] neighbourhoodCounts = EpsilonNeighbourCounter.computeNeighbourhoodDensities(
            source, envs, candidatePoints, eps, distanceMetric, chunkSize);

        // Step 3: core points, in ascending candidate order — the order that fixes cluster ids.
        List<double[]> cores = new ArrayList<>();
        double coreDegreeSum = 0.0;
        for (int i = 0; i < candidatePoints.length; i++) {
            if (neighbourhoodCounts[i] >= minPts) {
                cores.add(candidatePoints[i]);
                coreDegreeSum += neighbourhoodCounts[i];
            }
        }
        double[][] corePoints = cores.toArray(new double[0][]);
        logger.info("dbscanpp: {} of {} candidates are core points", corePoints.length, candidatePoints.length);

        if (corePoints.length == 0) {
            logger.warn("dbscanpp: no core point found (eps={}, minPts={}) — every point is noise",
                eps, minPts);
            return new CoreLabelModel(new double[0][], new int[0], eps, distanceMetric, requireWithinEps);
        }

        // Step 3b: eps-graph among the core points + connected components. The scan is m²/2
        // distance computations, so it goes to the cluster once m makes that worth a job; the
        // union-find itself always stays on the driver (see EpsilonGraphComponents).
        //
        // Step 2's counts are exact eps-degrees against the full data, so scaling them by m/n
        // estimates the degree INSIDE the core set for free: (Σ counts / m) · (m / n) = Σ counts / n.
        // It only sizes the log line here — unlike Spark, this path needs no edge-count budget.
        double expectedCoreDegree = datasetSize == 0L ? 0.0 : coreDegreeSum / datasetSize;
        int[] clusterLabels = EpsilonGraphComponents.compute(
            corePoints, eps, distanceMetric, envs, parallelism, expectedCoreDegree);
        logger.info("dbscanpp: {} clusters", (int) java.util.Arrays.stream(clusterLabels).distinct().count());

        return new CoreLabelModel(corePoints, clusterLabels, eps, distanceMetric, requireWithinEps);
    }
}
