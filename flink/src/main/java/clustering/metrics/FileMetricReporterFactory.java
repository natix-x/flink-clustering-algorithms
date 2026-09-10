package clustering.metrics;

import org.apache.flink.metrics.MetricConfig;
import org.apache.flink.metrics.reporter.MetricReporter;
import org.apache.flink.metrics.reporter.MetricReporterFactory;

import java.util.Properties;

/**
 * Factory for {@link FileMetricReporter}.
 * Expected Flink config key: metrics.reporter.file.factory.class
 */
public class FileMetricReporterFactory implements MetricReporterFactory {

    @Override
    public MetricReporter createMetricReporter(Properties properties) {
        FileMetricReporter reporter = new FileMetricReporter();

        MetricConfig metricConfig = new MetricConfig();
        metricConfig.putAll(properties);

        reporter.open(metricConfig);

        return reporter;
    }
}
