package clustering.core;


import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import java.io.Serializable;

/** Produces a fresh, BATCH-configured Flink environment per call.
 *
 *  A fresh env per Flink action keeps independent jobs (count, per-iteration
 *  reduce, eval) from accumulating transformations. Locally this is a new
 *  MiniCluster env each time; on a real cluster, multi-job-per-submission needs
 *  validation (see FlinkClusteringJob). */
@FunctionalInterface
public interface EnvFactory extends Serializable {
    StreamExecutionEnvironment newEnv();
}
