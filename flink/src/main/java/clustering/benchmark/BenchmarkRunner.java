package clustering.benchmark;

import clustering.benchmark.config.ClusterProfile;
import clustering.benchmark.config.RunConfig;
import clustering.benchmark.framework.FlinkClusteringJob;
import clustering.benchmark.metrics.RunResult;

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

    public static void main(String[] args) throws Exception {
        String configPath = parseConfigPath(args);
        RunConfig config = RunConfig.fromFile(configPath);

        ClusterProfile profile = config.resolveProfile();
        String outputDir = config.resolveOutputDir(profile);

        RunResult result = new FlinkClusteringJob().run(config, profile);
        Path path = RunResult.writeToDir(result, outputDir);

        System.out.println("[BenchmarkRunner] runId=" + result.runId + " status=" + result.status
            + " algo=" + result.algorithm + " fitMs=" + result.fitDurationMs
            + " totalMs=" + result.totalDurationMs + " nRows=" + result.nRows
            + " silhouette=" + result.silhouette + " -> " + path);

        if (!"ok".equals(result.status)) {
            System.exit(2);
        }
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
                    System.out.println(usage());
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
        System.err.println(msg + "\n" + usage());
        System.exit(64);
    }

    private static String usage() {
        return "Usage: flink run flink-clustering-benchmark.jar --config <path>\n\n"
            + "  --config  Path to a per-run JSON config (required)";
    }

    private BenchmarkRunner() {}
}