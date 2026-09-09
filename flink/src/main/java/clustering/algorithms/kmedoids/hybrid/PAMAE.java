package clustering.algorithms.kmedoids.hybrid;

import clustering.algorithms.kmedoids.local.DriverLocalKMedoids;
import clustering.core.Clusterer;
import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * PAMAE (Song, Jin, Kim &amp; Lee, KDD 2017) — the established parallel k-medoids algorithm, and the
 * only medoid entry whose optimisation sees the entire data.
 *
 * <p>Phase I is {@link CLARA}; phase II is {@code refineIters} Voronoi updates of the winner over
 * the whole dataset. So {@code pamae} and {@code clara} differ by exactly one phase — a controlled
 * experiment rather than two unrelated implementations — and on Flink they are literally the same
 * phase machine ({@link MedoidSearch}) run with refinement switched on.
 *
 * <p>All of it is ONE Flink job, including the candidate pool, which is drawn in the same pass that
 * draws CLARA's samples. It used to be three jobs and therefore three reads of the source; see
 * {@link MedoidSearch} for what that cost and why Spark's PAMAE keeps the three-job shape.
 *
 * @param refinementIterations phase-II iterations; the paper uses 1
 * @param poolSize             phase-II candidate pool — the accuracy-vs-cost knob
 */
public class PAMAE implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(PAMAE.class);

    private final String localSolverName;
    private final MedoidSearch search;

    public PAMAE(int targetK, int numSamples, int sampleSize, int maxLocalIterations,
                 int refinementIterations, int poolSize, DistanceMetric distanceMetric,
                 String localSolverName, long seed) {

        if (refinementIterations < 1) {
            throw new IllegalArgumentException("refineIters must be >= 1, got " + refinementIterations);
        }

        DriverLocalKMedoids localSolver =
            DriverLocalKMedoids.fromName(localSolverName, targetK, maxLocalIterations, distanceMetric, seed);
        this.localSolverName = localSolverName;
        this.search = new MedoidSearch(targetK, numSamples, sampleSize, poolSize, refinementIterations,
                                       distanceMetric, localSolver, seed);
    }

    @Override
    public Model fit(PointSource source, EnvFactory envFactory, int parallelism) {
        MedoidSearch.Result result = search.run(source, envFactory);

        double improvement =
            (result.seedingCost - result.finalCost) / Math.max(result.seedingCost, 1e-12) * 100.0;

        // What the "entire data" half bought is the paper's headline number, so every run records it.
        logger.info("pamae: seedingCost={} refinedCost={} improvement={}% iterations={} poolSize={} solver={}",
            String.format("%.4f", result.seedingCost),
            String.format("%.4f", result.finalCost),
            String.format("%.2f", improvement),
            result.refinementRounds,
            result.poolSize,
            localSolverName);

        return result.model;
    }
}
