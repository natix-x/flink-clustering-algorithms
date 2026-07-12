package clustering.metrics;

import clustering.benchmark.metrics.BenchmarkListener;
import clustering.benchmark.metrics.MetricsFile;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStreamUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** Probe: the file metric reporter loads in a local MiniCluster, writes per-process files,
 *  and {@link MetricsFile} reads non-zero engine metrics back. */
class ReporterProbeTest {

    @Test
    void reporterWritesReadableMetrics() throws Exception {
        Path base = Paths.get(System.getProperty("java.io.tmpdir"), "reporter-probe.txt");
        System.setProperty("clustering.metrics.file", base.toString());
        MetricsFile.reset();

        Configuration conf = new Configuration();
        conf.setString("metrics.reporter.file.factory.class",
            "clustering.metrics.FileMetricReporterFactory");
        conf.setString("metrics.reporter.file.path", base.toString());
        conf.setString("metrics.reporter.file.interval", "1 SECONDS");

        StreamExecutionEnvironment env = StreamExecutionEnvironment.createLocalEnvironment(2, conf);
        env.setParallelism(2);

        Iterator<Long> it = DataStreamUtils.collect(
            env.fromSequence(1, 5_000_000).map(x -> x).keyBy(x -> 0L).sum(0));
        while (it.hasNext()) {
            it.next();
        }
        Thread.sleep(1500);  // let report() ticks fire

        BenchmarkListener.Snapshot s = MetricsFile.read();
        assertTrue(s.taskCount > 0, "expected some task instances");
        assertTrue(s.peakExecutorMemoryBytes > 0, "expected TM heap metric");
    }
}
