package clustering.benchmark.metrics;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads and aggregates plain-text files written by Flink's FileMetricReporter.
 * Combines TaskManager and JobManager metrics into a single BenchmarkListener.Snapshot.
 */
public final class MetricsFile {

    private MetricsFile() {}

    /**
     * Resolves the base path for reporter files using the CLUSTERING_METRICS_FILE env variable,
     * the clustering.metrics.file system property, or a temp file fallback.
     */
    public static String path() {
        String p = System.getenv("CLUSTERING_METRICS_FILE");
        if (p == null || p.isEmpty()) {
            p = System.getProperty("clustering.metrics.file",
                System.getProperty("java.io.tmpdir") + "/flink-metrics.txt");
        }
        return p;
    }

    /**
     * Checks if the metrics file path was set explicitly via environment or system properties,
     * rather than falling back to the local temp directory.
     */
    public static boolean isExplicitlyConfigured() {
        String env = System.getenv("CLUSTERING_METRICS_FILE");
        if (env != null && !env.isEmpty()) {
            return true;
        }
        return System.getProperty("clustering.metrics.file") != null;
    }

    /**
     * Deletes all existing reporter and time-series files for a clean benchmark run.
     */
    public static void reset() {
        List<Path> stale = new ArrayList<>(reporterFiles());
        stale.addAll(seriesFiles());
        for (Path f : stale) {
            try {
                Files.deleteIfExists(f);
            } catch (IOException ignore) {
                // best-effort
            }
        }
    }

    private static List<Path> reporterFiles() {
        List<Path> out = new ArrayList<>();
        Path base = Paths.get(path());
        Path dir = base.getParent() == null ? Paths.get(".") : base.getParent();
        String prefix = base.getFileName().toString() + ".";

        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, prefix + "*")) {
            for (Path p : ds) {
                String name = p.getFileName().toString();
                if (!name.endsWith(".writing") && !name.endsWith(".series")) {
                    out.add(p);
                }
            }
        } catch (IOException ignore) {
            // dir missing -> no files
        }
        return out;
    }

    /**
     * Parses and aggregates all reporter files into a Snapshot.
     */
    public static BenchmarkListener.Snapshot read() {
        BenchmarkListener.Snapshot s = new BenchmarkListener.Snapshot();
        long heap = 0, heapMax = 0;
        long driverHeapMax = 0;
        long direct = 0, directMax = 0, driverDirectMax = 0;

        for (Path file : reporterFiles()) {
            List<String> lines;
            try {
                lines = Files.readAllLines(file, StandardCharsets.UTF_8);
            } catch (IOException e) {
                continue;
            }

            for (String line : lines) {
                int tab = line.lastIndexOf('\t');
                if (tab < 0) {
                    continue;
                }

                String id = line.substring(0, tab);
                long v;
                try {
                    v = Long.parseLong(line.substring(tab + 1).trim());
                } catch (NumberFormatException e) {
                    continue;
                }

                boolean tm = id.contains(".taskmanager.");
                boolean jm = id.contains(".jobmanager.");

                if (tm && id.endsWith(".Status.JVM.CPU.Time")) {
                    s.cpuByProcess.merge(processOf(id), v, Math::max);
                } else if (tm && id.endsWith(".Status.JVM.Memory.Heap.Used.TruePeak")) {
                    heap += v;
                    heapMax = Math.max(heapMax, v);
                } else if (tm && id.endsWith(".Status.JVM.Memory.Heap.Used.ByteSeconds")) {
                    s.heapSecondsByProcess.merge(processOf(id), v, Math::max);
                } else if (tm && id.endsWith(".Status.JVM.Memory.Direct.Used")) {
                    direct += v;
                    directMax = Math.max(directMax, v);
                } else if (tm && id.contains("GarbageCollector") && id.endsWith(".Time")) {
                    s.gcByProcess.merge(processOf(id), v, Long::sum);
                } else if (jm && id.endsWith(".Status.JVM.CPU.Time")) {
                    s.driverCpuByProcess.merge(processOf(id), v, Math::max);
                } else if (jm && id.endsWith(".Status.JVM.Memory.Heap.Used.TruePeak")) {
                    driverHeapMax = Math.max(driverHeapMax, v);
                } else if (jm && id.endsWith(".Status.JVM.Memory.Heap.Used.ByteSeconds")) {
                    s.driverHeapSecondsByProcess.merge(processOf(id), v, Math::max);
                } else if (jm && id.endsWith(".Status.JVM.Memory.Direct.Used")) {
                    driverDirectMax = Math.max(driverDirectMax, v);
                } else if (jm && id.contains("GarbageCollector") && id.endsWith(".Time")) {
                    s.driverGcByProcess.merge(processOf(id), v, Long::sum);
                }
            }
        }

        s.peakExecutorMemoryBytes = heapMax;
        s.totalExecutorMemoryBytes = heap;
        s.peakDirectMemoryBytes = directMax;
        s.totalDirectMemoryBytes = direct;
        s.driverPeakHeapBytes = driverHeapMax;
        s.driverPeakDirectMemoryBytes = driverDirectMax;

        s.recomputeTotals();
        return s;
    }

    /**
     * Extracts the process identifier from a metric ID to allow proper per-process delta calculations.
     */
    private static String processOf(String id) {
        int marker = id.indexOf("Status.JVM.");
        return marker < 0 ? id : id.substring(0, marker);
    }

    private static List<Path> seriesFiles() {
        List<Path> out = new ArrayList<>();
        Path base = Paths.get(path());
        Path dir = base.getParent() == null ? Paths.get(".") : base.getParent();
        String prefix = base.getFileName().toString() + ".";

        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, prefix + "*.series")) {
            for (Path p : ds) {
                out.add(p);
            }
        } catch (IOException ignore) {
            // dir missing -> no files
        }
        return out;
    }

    /**
     * Calculates heap and direct memory peaks within a specific wall-clock window
     * using the time-series files.
     */
    public static BenchmarkListener.PhasePeaks windowPeaks(long fromMs, long toMs) {
        java.util.Map<String, Long> maxById = new java.util.HashMap<>();

        for (Path file : seriesFiles()) {
            List<String> lines;
            try {
                lines = Files.readAllLines(file, StandardCharsets.UTF_8);
            } catch (IOException e) {
                continue;
            }

            for (String line : lines) {
                int firstTab = line.indexOf('\t');
                int lastTab = line.lastIndexOf('\t');

                if (firstTab < 0 || lastTab <= firstTab) {
                    continue;
                }

                long ts;
                long v;
                try {
                    ts = Long.parseLong(line.substring(0, firstTab).trim());
                    v = Long.parseLong(line.substring(lastTab + 1).trim());
                } catch (NumberFormatException e) {
                    continue;
                }

                if (ts < fromMs || ts > toMs) {
                    continue;
                }

                String id = line.substring(firstTab + 1, lastTab);
                maxById.merge(id, v, Math::max);
            }
        }

        BenchmarkListener.PhasePeaks p = new BenchmarkListener.PhasePeaks();

        for (java.util.Map.Entry<String, Long> e : maxById.entrySet()) {
            String id = e.getKey();
            long v = e.getValue();

            boolean tm = id.contains(".taskmanager.");
            boolean jm = id.contains(".jobmanager.");
            boolean heap = id.endsWith("Status.JVM.Memory.Heap.Used.WindowPeak");
            boolean direct = id.endsWith("Status.JVM.Memory.Direct.Used.Current");

            if (tm && heap) {
                p.maxWorkerHeapBytes = Math.max(p.maxWorkerHeapBytes, v);
                p.totalWorkerHeapBytes += v;
            } else if (tm && direct) {
                p.maxWorkerDirectBytes = Math.max(p.maxWorkerDirectBytes, v);
                p.totalWorkerDirectBytes += v;
            } else if (jm && heap) {
                p.driverHeapBytes = Math.max(p.driverHeapBytes, v);
            } else if (jm && direct) {
                p.driverDirectBytes = Math.max(p.driverDirectBytes, v);
            }
        }
        return p;
    }
}
