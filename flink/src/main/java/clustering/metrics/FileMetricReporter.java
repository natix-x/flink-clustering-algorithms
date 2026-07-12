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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/** Plain-text Flink metric reporter for the benchmark: it writes the built-in Flink metrics
 *  we care about (network bytes, records, JVM GC/CPU/heap) to a text file, so the driver
 *  ({@code BenchmarkRunner}) can read them back into the {@code RunResult} — WITHOUT touching
 *  any algorithm code (no per-operator accumulators) and WITHOUT the driver polling REST.
 *
 *  Why a custom file reporter (not Prometheus/Slf4j): the benchmark jobs are short BATCH
 *  jobs, so a PULL reporter (Prometheus, ~15s scrape) misses them; this flushes every
 *  {@link #report()} tick and on {@link #close()}, capturing even sub-second jobs. Output is
 *  plain text (no Jackson) so the reporter has zero deps beyond flink-metrics-core and can be
 *  dropped into {@code $FLINK_HOME/lib} on the cluster.
 *
 *  <h3>One file per reporter instance</h3>
 *  A MiniCluster (and a real cluster) runs a SEPARATE reporter instance per process
 *  (JobManager + each TaskManager). They would clobber a shared file, so each instance writes
 *  to {@code <path>.<uuid>}; {@link clustering.benchmark.metrics.MetricsFile} globs and
 *  aggregates all of them. On a cluster point {@code path} at the shared filesystem so every
 *  node's file is visible to the driver.
 *
 *  <h3>Lifetime &amp; why we track the max</h3>
 *  One standalone cluster runs exactly one RunConfig, so the files hold only that run. Both
 *  task-scoped counters (numBytesIn, ...) and TaskManager JVM gauges (CPU/GC/heap) are torn
 *  down before the reporter's final {@link #close()} flush — tasks when their job ends, TM
 *  gauges when the cluster stops. A naive "write current live metrics" loses them, because the
 *  last flush sees an empty registry. So we record the MAX value ever observed per metric id
 *  (in every {@link #report()} tick and again on removal) in {@link #peak} and always write
 *  that. All kept metrics are either monotonic (numBytes/numRecords counters, CPU.Time,
 *  GC.Time — max == final) or want-peak (Heap.Used — max == peak), so max is correct.
 *
 *  Config (flink-conf.yaml / config.yaml):
 *  <pre>
 *  metrics.reporter.file.factory.class: clustering.metrics.FileMetricReporterFactory
 *  metrics.reporter.file.path: /path/to/flink-metrics.txt
 *  metrics.reporter.file.interval: 1 SECONDS
 *  </pre> */
public class FileMetricReporter implements MetricReporter, Scheduled {

    private static final Logger logger = LoggerFactory.getLogger(FileMetricReporter.class);

    /** Identifier fragments we keep (matched against the FULL metric identifier). */
    private static final String[] KEEP = {
        "numBytesIn", "numBytesOut", "numRecordsIn", "numRecordsOut",
        "GarbageCollector", "Status.JVM.CPU.Time", "Status.JVM.Memory.Heap.Used"
    };

    private Path path;
    private final Map<String, Metric> live = new ConcurrentHashMap<>();
    /** Max value ever observed per kept metric id — survives task/TM teardown. */
    private final Map<String, Long> peak = new ConcurrentHashMap<>();

    @Override
    public void open(MetricConfig config) {
        String base = config.getString("path", System.getProperty("clustering.metrics.file",
            System.getProperty("java.io.tmpdir") + "/flink-metrics.txt"));
        this.path = Paths.get(base + "." + UUID.randomUUID());
    }

    @Override
    public void close() {
        report();  // final flush
    }

    @Override
    public void notifyOfAddedMetric(Metric metric, String metricName, MetricGroup group) {
        if (!(metric instanceof Counter) && !(metric instanceof Gauge)) {
            return;
        }
        String id = group.getMetricIdentifier(metricName);
        if (keep(id)) {
            live.put(id, metric);
        }
    }

    @Override
    public void notifyOfRemovedMetric(Metric metric, String metricName, MetricGroup group) {
        String id = group.getMetricIdentifier(metricName);
        if (!keep(id)) {
            return;
        }
        live.remove(id);
        // Preserve the final value of any torn-down metric (counter OR gauge) before it
        // disappears from the registry — tasks tear down per job, TM gauges at cluster stop.
        Long v = value(metric);
        if (v != null) {
            peak.merge(id, v, Math::max);
        }
    }

    @Override
    public synchronized void report() {
        if (path == null) {
            return;
        }
        // Fold every currently-live metric into the peak map (max), then write the peak map —
        // so values survive the teardown that precedes the final close() flush.
        for (Map.Entry<String, Metric> e : live.entrySet()) {
            Long v = value(e.getValue());
            if (v != null) {
                peak.merge(e.getKey(), v, Math::max);
            }
        }
        logger.debug("PROBE report path={} live={} peak={}", path.getFileName(), live.size(), peak.size());
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Long> e : peak.entrySet()) {
            sb.append(e.getKey()).append('\t').append(e.getValue()).append('\n');
        }
        try {
            Path tmp = Paths.get(path.toString() + ".writing");
            try (Writer w = Files.newBufferedWriter(tmp, StandardCharsets.UTF_8)) {
                w.write(sb.toString());
            }
            Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException ignore) {
            // best-effort; a missed flush just leaves slightly stale metrics
        }
    }

    private static Long value(Metric m) {
        if (m instanceof Counter) {
            return ((Counter) m).getCount();
        }
        if (m instanceof Gauge) {
            Object v = ((Gauge<?>) m).getValue();
            if (v instanceof Number) {
                return ((Number) v).longValue();
            }
        }
        return null;
    }

    private static boolean keep(String id) {
        for (String k : KEEP) {
            if (id.contains(k)) {
                return true;
            }
        }
        return false;
    }
}
