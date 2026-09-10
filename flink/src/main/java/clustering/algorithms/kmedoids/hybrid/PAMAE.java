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
 * PAMAE algorithm.
 * Extends CLARA by performing Voronoi updates (refinement iterations) of the winning
 * medoid set over the entire dataset to improve accuracy.
 */
public class PAMAE implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(PAMAE.class);

    private final String localSolverName;
    private final MedoidSearch search;

    public PAMAE(
            int targetK,
            int numSamples,
            int sampleSize,
            int maxLocalIterations,
            int refinementIterations,
            int poolSize,
            DistanceMetric distanceMetric,
            String localSolverName,
            long seed
    ) {
        if (refinementIterations < 1) {
            throw new IllegalArgumentException("refineIters must be >= 1, got " + refinementIterations);
        }

        DriverLocalKMedoids localSolver = DriverLocalKMedoids.fromName(
            localSolverName, targetK, maxLocalIterations, distanceMetric, seed
        );
        this.localSolverName = localSolverName;
        this.search = new MedoidSearch(
            targetK, numSamples, sampleSize, poolSize, refinementIterations,
            distanceMetric, localSolver, seed
        );
    }

    @Override
    public Model fit(PointSource source, EnvFactory envFactory, int parallelism) {
        MedoidSearch.Result result = search.run(source, envFactory);

        double improvement =
            (result.seedingCost - result.finalCost) / Math.max(result.seedingCost, 1e-12) * 100.0;

        logger.info("pamae: seedingCost={} refinedCost={} improvement={}% iterations={} poolSize={} solver={}",
            String.format("%.4f", result.seedingCost),
            String.format("%.4f", result.finalCost),
            String.format("%.2f", improvement),
            result.refinementRounds,
            result.poolSize,
            localSolverName
        );

        return result.model;
    }
}
