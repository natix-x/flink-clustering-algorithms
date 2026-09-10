package clustering.metrics;

import org.apache.flink.metrics.Counter;
import org.apache.flink.metrics.Gauge;
import org.apache.flink.metrics.Metric;
import org.apache.flink.metrics.MetricConfig;
import org.apache.flink.metrics.MetricGroup;
import org.apache.flink.metrics.reporter.MetricReporter;
import org.apache.flink.metrics.reporter.Scheduled;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Writer;
import java.lang.management.BufferPoolMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Plain-text Flink metric reporter.
 * Writes built-in Flink metrics (record counts, JVM GC/CPU/heap) to text files.
 *
 * Features:
 * - Maintains peak values across job teardowns.
 * - Tracks gap-free high-water marks for Heap memory using JVM MXBeans.
 * - Tracks Netty's off-heap direct buffer usage.
 * - Maintains an append-only time-series file for per-phase peak extraction.
 * - Calculates a Riemann sum (ByteSeconds) for total memory consumption over time.
 */
public class FileMetricReporter implements MetricReporter, Scheduled {

    private static final Logger LOG = LoggerFactory.getLogger(FileMetricReporter.class);

    private static final String HEAP_USED_SUFFIX = "Status.JVM.Memory.Heap.Used";
    private static final String DIRECT_USED_SUFFIX = "Status.JVM.Memory.Direct.Used";

    private static final String[] METRICS_TO_TRACK = {
        "numRecordsIn", "numRecordsOut",
        "GarbageCollector", "Status.JVM.CPU.Time", HEAP_USED_SUFFIX
    };

    private Path latestMetricsFile;
    private Path timeSeriesFile;

    private final Map<String, Metric> activeMetrics = new ConcurrentHashMap<>();
    private final Map<String, Long> peakMetricValues = new ConcurrentHashMap<>();
    private final Map<String, Double> heapByteSecondsIntegral = new ConcurrentHashMap<>();

    private long lastReportTimestampNanos = -1L;

    private final List<MemoryPoolMXBean> trackableHeapPools = new ArrayList<>();
    private final List<BufferPoolMXBean> directBufferPools = new ArrayList<>();

    @Override
    public void open(MetricConfig config) {
        String basePath = config.getString("path", System.getProperty("clustering.metrics.file",
            System.getProperty("java.io.tmpdir") + "/flink-metrics.txt"));

        this.latestMetricsFile = Paths.get(basePath + "." + UUID.randomUUID());
        this.timeSeriesFile = Paths.get(this.latestMetricsFile + ".series");
        this.lastReportTimestampNanos = System.nanoTime();

        for (MemoryPoolMXBean pool : ManagementFactory.getMemoryPoolMXBeans()) {
            if (pool.getType() == MemoryType.HEAP && pool.getPeakUsage() != null) {
                try {
                    pool.resetPeakUsage();
                    trackableHeapPools.add(pool);
                } catch (UnsupportedOperationException ignore) {
                    // Skip pools that don't support peak reset
                }
            }
        }

        for (BufferPoolMXBean pool : ManagementFactory.getPlatformMXBeans(BufferPoolMXBean.class)) {
            if ("direct".equals(pool.getName())) {
                directBufferPools.add(pool);
            }
        }
    }

    @Override
    public void close() {
        report(); // Final flush
    }

    @Override
    public void notifyOfAddedMetric(Metric metric, String metricName, MetricGroup group) {
        if (!(metric instanceof Counter) && !(metric instanceof Gauge)) {
            return;
        }
        String metricId = group.getMetricIdentifier(metricName);
        if (shouldTrackMetric(metricId)) {
            activeMetrics.put(metricId, metric);
        }
    }

    @Override
    public void notifyOfRemovedMetric(Metric metric, String metricName, MetricGroup group) {
        String metricId = group.getMetricIdentifier(metricName);
        if (!shouldTrackMetric(metricId)) {
            return;
        }
        activeMetrics.remove(metricId);

        Long metricValue = extractLongValue(metric);
        if (metricValue != null) {
            peakMetricValues.merge(metricId, metricValue, Math::max);
        }
    }

    @Override
    public synchronized void report() {
        if (latestMetricsFile == null) {
            return;
        }

        long currentNanos = System.nanoTime();
        double elapsedSeconds = lastReportTimestampNanos < 0 ? 0.0 : (currentNanos - lastReportTimestampNanos) / 1e9;
        lastReportTimestampNanos = currentNanos;

        for (Map.Entry<String, Metric> entry : activeMetrics.entrySet()) {
            String metricId = entry.getKey();
            Long value = extractLongValue(entry.getValue());

            if (value != null) {
                peakMetricValues.merge(metricId, value, Math::max);
                if (elapsedSeconds > 0 && metricId.contains(HEAP_USED_SUFFIX)) {
                    heapByteSecondsIntegral.merge(metricId, value * elapsedSeconds, Double::sum);
                }
            }
        }

        long currentHeapPeak = getAndResetHeapPeak();
        if (!trackableHeapPools.isEmpty()) {
            for (String metricId : new ArrayList<>(peakMetricValues.keySet())) {
                if (metricId.endsWith(HEAP_USED_SUFFIX)) {
                    peakMetricValues.merge(metricId + ".TruePeak", currentHeapPeak, Math::max);
                }
            }
        }

        if (!directBufferPools.isEmpty()) {
            long currentDirectUsed = 0L;
            for (BufferPoolMXBean pool : directBufferPools) {
                currentDirectUsed += pool.getMemoryUsed();
            }

            for (String metricId : new ArrayList<>(peakMetricValues.keySet())) {
                if (metricId.endsWith(HEAP_USED_SUFFIX)) {
                    String scopePrefix = metricId.substring(0, metricId.length() - HEAP_USED_SUFFIX.length());
                    peakMetricValues.merge(scopePrefix + DIRECT_USED_SUFFIX, currentDirectUsed, Math::max);
                }
            }
        }

        appendTimeSeriesReadings(currentHeapPeak);
        LOG.debug("PROBE report path={} live={} peak={}", latestMetricsFile.getFileName(), activeMetrics.size(), peakMetricValues.size());

        flushLatestMetricsToFile();
    }

    private void flushLatestMetricsToFile() {
        StringBuilder fileContent = new StringBuilder();

        for (Map.Entry<String, Long> entry : peakMetricValues.entrySet()) {
            fileContent.append(entry.getKey()).append('\t').append(entry.getValue()).append('\n');
        }
        for (Map.Entry<String, Double> entry : heapByteSecondsIntegral.entrySet()) {
            fileContent.append(entry.getKey()).append(".ByteSeconds\t").append(Math.round(entry.getValue())).append('\n');
        }

        try {
            Path tempFile = Paths.get(latestMetricsFile.toString() + ".writing");
            try (Writer writer = Files.newBufferedWriter(tempFile, StandardCharsets.UTF_8)) {
                writer.write(fileContent.toString());
            }
            Files.move(tempFile, latestMetricsFile, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException ignore) {
            // Best-effort flush
        }
    }

    private void appendTimeSeriesReadings(long currentHeapPeak) {
        if (timeSeriesFile == null || trackableHeapPools.isEmpty()) {
            return;
        }

        List<String> reporterPrefixes = new ArrayList<>();
        for (String metricId : peakMetricValues.keySet()) {
            if (metricId.endsWith(HEAP_USED_SUFFIX)) {
                reporterPrefixes.add(metricId.substring(0, metricId.length() - HEAP_USED_SUFFIX.length()));
            }
        }

        if (reporterPrefixes.isEmpty()) {
            return;
        }

        long currentDirectUsed = 0L;
        for (BufferPoolMXBean pool : directBufferPools) {
            currentDirectUsed += pool.getMemoryUsed();
        }

        long timestampMillis = System.currentTimeMillis();
        StringBuilder seriesLines = new StringBuilder();

        for (String prefix : reporterPrefixes) {
            seriesLines.append(timestampMillis).append('\t').append(prefix)
                .append(HEAP_USED_SUFFIX).append(".WindowPeak\t").append(currentHeapPeak).append('\n');
            seriesLines.append(timestampMillis).append('\t').append(prefix)
                .append(DIRECT_USED_SUFFIX).append(".Current\t").append(currentDirectUsed).append('\n');
        }

        try {
            Files.write(timeSeriesFile, seriesLines.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException ignore) {
            // Best-effort append
        }
    }

    private long getAndResetHeapPeak() {
        long totalPeak = 0L;
        for (MemoryPoolMXBean pool : trackableHeapPools) {
            totalPeak += pool.getPeakUsage().getUsed();
            pool.resetPeakUsage();
        }
        return totalPeak;
    }

    private static Long extractLongValue(Metric metric) {
        if (metric instanceof Counter) {
            return ((Counter) metric).getCount();
        }
        if (metric instanceof Gauge) {
            Object value = ((Gauge<?>) metric).getValue();
            if (value instanceof Number) {
                return ((Number) value).longValue();
            }
        }
        return null;
    }

    private static boolean shouldTrackMetric(String metricId) {
        for (String targetMetric : METRICS_TO_TRACK) {
            if (metricId.contains(targetMetric)) {
                return true;
            }
        }
        return false;
    }
}
