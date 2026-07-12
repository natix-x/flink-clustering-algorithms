package clustering.benchmark;

import clustering.benchmark.config.ClusterProfile;
import clustering.benchmark.config.RunConfig;
import clustering.benchmark.metrics.RunResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

/** Entry point for one benchmark run.
 *
 *  Usage: {@code flink run flink-clustering-benchmark.jar --config <path-to-run.json>}
 *
 *  Exactly one config file produces exactly one JSON result file under
 *  {@code <outputDir>/<runId>.json}. Failures still write a result (status
 *  "failed" + errorMessage) so array jobs never silently lose runs. Java mirror
 *  of the Spark {@code BenchmarkRunner}. */
public final class BenchmarkRunner {

    private static final Logger logger = LoggerFactory.getLogger(BenchmarkRunner.class);

    public static void main(String[] args) throws Exception {
        String configPath = parseConfigPath(args);
        RunConfig config = RunConfig.fromFile(configPath);

        ClusterProfile profile = config.resolveProfile();
        String outputDir = config.resolveOutputDir();

        RunResult result = new FlinkClusteringJob().run(config, profile);
        Path path = RunResult.writeToDir(result, outputDir);

        logger.info("runId={} status={} algo={} fitMs={} totalMs={} nRows={} silhouette={} -> {}",
            result.runId, result.status, result.algorithm, result.fitDurationMs,
            result.totalDurationMs, result.nRows, result.silhouette, path);

        // Exit explicitly so Flink's JVM shutdown hooks run while the (exec:java) classloader
        // is still open — a bare return lets exec tear the loader down first, which surfaces a
        // benign NoClassDefFoundError from MemoryExecutionGraphInfoStore.close(). The code also
        // propagates to the caller (SLURM): 0 = ok, 2 = failed.
        System.exit("ok".equals(result.status) ? 0 : 2);
    }

    private static String parseConfigPath(String[] args) {
        String configPath = null;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--config":
                    if (i + 1 >= args.length) {
                        fail("--config requires a value");
                    }
                    configPath = args[++i];
                    break;
                case "--help":
                case "-h":
                    logger.info(usage());
                    System.exit(0);
                    break;
                default:
                    fail("Unknown argument: " + args[i]);
            }
        }
        if (configPath == null) {
            fail("--config is required");
        }
        return configPath;
    }

    private static void fail(String msg) {
        logger.error("{}\n{}", msg, usage());
        System.exit(64);
    }

    private static String usage() {
        return "Usage: flink run flink-clustering-benchmark.jar --config <path>\n\n"
            + "  --config  Path to a per-run JSON config (required)";
    }

    private BenchmarkRunner() {}
}
