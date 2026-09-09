package clustering.core;

import clustering.algorithms.kmeans.CentroidIteration;
import java.io.Serializable;

/** Fits a {@link Model} to a distributed point source.
 *
 *  Implementations decide how to use Flink: collect-to-driver (PAM, FastPAM), one job per
 *  step (CLARA), or the Flink ML bounded-iteration framework as a single job (everything
 *  iterative — see {@link clustering.algorithms.kmeans.CentroidIteration}). The
 *  {@link EnvFactory} hands out fresh environments and {@code parallelism} is the configured
 *  data/compute parallelism. */
public interface Clusterer extends Serializable {
    Model fit(PointSource source, EnvFactory envs, int parallelism);
}
