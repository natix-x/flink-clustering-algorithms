package clustering.metrics;

import org.apache.flink.metrics.MetricConfig;
import org.apache.flink.metrics.reporter.MetricReporter;
import org.apache.flink.metrics.reporter.MetricReporterFactory;
import java.util.Properties;

/** Factory for {@link FileMetricReporter}. Referenced from the Flink config:
 *  {@code metrics.reporter.file.factory.class: clustering.metrics.FileMetricReporterFactory}. */
public class FileMetricReporterFactory implements MetricReporterFactory {

    @Override
    public MetricReporter createMetricReporter(Properties properties) {
        FileMetricReporter reporter = new FileMetricReporter();
        MetricConfig config = new MetricConfig();
        config.putAll(properties);
        reporter.open(config);
        return reporter;
    }
}
