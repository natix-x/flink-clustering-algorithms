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
 * CLARA (Clustering LARge Applications) algorithm.
 * Runs an exact k-medoids solver on small random samples and selects the candidate
 * medoid set that yields the lowest cost across the entire dataset.
 */
public class CLARA implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(CLARA.class);

    private final String localSolverName;
    private final int numSamples;
    private final int sampleSize;
    private final MedoidSearch search;

    public CLARA(
            int targetK,
            int numSamples,
            int sampleSize,
            int maxIterations,
            DistanceMetric distanceMetric,
            String localSolverName,
            long seed
    ) {
        DriverLocalKMedoids localSolver = DriverLocalKMedoids.fromName(
            localSolverName, targetK, maxIterations, distanceMetric, seed
        );
        this.localSolverName = localSolverName;
        this.numSamples = numSamples;
        this.sampleSize = sampleSize;
        this.search = new MedoidSearch(
            targetK, numSamples, sampleSize, 0, 0, distanceMetric, localSolver, seed
        );
    }

    @Override
    public Model fit(PointSource source, EnvFactory envFactory, int parallelism) {
        MedoidSearch.Result result = search.run(source, envFactory);

        logger.info("clara: n={} samples={} sampleSize={} solver={} cost={}",
            result.totalPoints, numSamples, sampleSize, localSolverName, result.finalCost);

        return result.model;
    }
}
