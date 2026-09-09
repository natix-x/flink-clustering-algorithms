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
 * CLARA (Kaufman &amp; Rousseeuw 1990) — run an exact solver on small random samples, keep the
 * medoid set whose cost on the FULL dataset is lowest.
 *
 * <p>One Flink job: {@link MedoidSearch} draws all {@code numSamples} samples in its first round
 * and scores every candidate set in its second, both off the point cache the job's single read of
 * the source fills. The cost is therefore two passes whatever {@code numSamples} is — which is why
 * the separate {@code claraflip} entry (one sample per round) was folded away: batching the draw
 * and the scoring inside the cached iteration does the same distance work in 2 passes instead of
 * {@code numSamples + 1}.
 *
 * @param localSolverName exact solver run on each sample ({@code fastpam} | {@code fasterpam}) —
 *                        the knob Schubert &amp; Rousseeuw 2021 improve CLARA by: the sampling
 *                        scaffold is unchanged, only the solver inside it gets cheaper.
 */
public class CLARA implements Clusterer {

    private static final Logger logger = LoggerFactory.getLogger(CLARA.class);

    private final String localSolverName;
    private final int numSamples;
    private final int sampleSize;
    private final MedoidSearch search;

    public CLARA(int targetK, int numSamples, int sampleSize, int maxIterations, DistanceMetric distanceMetric,
                 String localSolverName, long seed) {
        DriverLocalKMedoids localSolver =
            DriverLocalKMedoids.fromName(localSolverName, targetK, maxIterations, distanceMetric, seed);
        this.localSolverName = localSolverName;
        this.numSamples = numSamples;
        this.sampleSize = sampleSize;
        this.search = new MedoidSearch(
            targetK, numSamples, sampleSize, 0, 0, distanceMetric, localSolver, seed);
    }

    @Override
    public Model fit(PointSource source, EnvFactory envFactory, int parallelism) {
        MedoidSearch.Result result = search.run(source, envFactory);
        logger.info("clara: n={} samples={} sampleSize={} solver={} cost={}",
            result.totalPoints, numSamples, sampleSize, localSolverName, result.finalCost);
        return result.model;
    }
}
