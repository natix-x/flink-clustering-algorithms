package clustering.core;

import java.io.Serializable;

/** Fits a {@link Model} to a distributed point source.
 *
 *  Implementations decide how to use Flink: collect-to-driver (impl=driverloop),
 *  one job per iteration (impl=jobiter), or the Flink ML iteration framework
 *  (impl=mliter). The {@link EnvFactory} hands out fresh environments and
 *  {@code parallelism} is the configured data/compute parallelism. */
public interface Clusterer extends Serializable {
    Model fit(PointSource source, EnvFactory envs, int parallelism);
}
