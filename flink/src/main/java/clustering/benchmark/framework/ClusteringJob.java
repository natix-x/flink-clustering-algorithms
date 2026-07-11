package clustering.benchmark.framework;

import clustering.benchmark.config.ClusterProfile;
import clustering.benchmark.config.RunConfig;
import clustering.benchmark.metrics.RunResult;

/** Framework-agnostic seam, one impl per engine. Java mirror of the Spark
 *  {@code clustering.benchmark.framework.ClusteringJob}. */
public interface ClusteringJob {
    String framework();
    RunResult run(RunConfig config, ClusterProfile profile);
}