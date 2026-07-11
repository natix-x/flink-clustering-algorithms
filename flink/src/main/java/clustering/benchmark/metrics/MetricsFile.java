package clustering.benchmark.metrics;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Reads the plain-text files written by {@code clustering.metrics.FileMetricReporter} (one
 *  per JobManager/TaskManager process, named {@code <path>.<uuid>}) and aggregates Flink's
 *  built-in metrics into a {@link BenchmarkListener.Snapshot}.
 *
 *  Each line is {@code <metric-identifier>\t<long value>}. Counters are SUMMED across task
 *  instances; TaskManager JVM gauges are summed across TMs (each is that TM's run-end value).
 *  Only {@code .taskmanager.}-scoped JVM metrics are counted (the TaskManager is the Flink
 *  analogue of a Spark executor).
 *
 *  <h3>Cross-framework comparability (Spark port)</h3>
 *  To match the Spark {@code RunResult} on the SAME measurement basis:
 *  <ul>
 *    <li>{@code inputBytes}  = bytes a SOURCE operator emits ({@code Source:*.numBytesOut}) —
 *        the Flink analogue of Spark {@code InputMetrics.bytesRead} (source read only, no
 *        shuffle/intermediate).</li>
 *    <li>{@code outputBytes} = bytes a SINK operator receives ({@code Sink:*.numBytesIn}) —
 *        the analogue of Spark {@code OutputMetrics.bytesWritten}.</li>
 *    <li>{@code shuffleReadBytes}/{@code shuffleWriteBytes} = total shuffle volume from the
 *        {@code Shuffle.Netty.Input} scope (local + remote), the Flink analogue of Spark
 *        {@code shuffleReadBytes}. Flink has no read/write split, so both report it. (Counting
 *        the loose {@code numBytesInRemote} would double-count an operator-level alias and
 *        miss local shuffle, which is why a single-JVM local run showed 0.)</li>
 *    <li>{@code peakExecutorMemoryBytes} = JVM heap used (analogue of Spark
 *        {@code JVMHeapMemory}). Spark's off-heap has no clean Flink analogue (Flink managed
 *        memory ≠ Spark off-heap), so heap-only is the comparable basis.</li>
 *  </ul> */
public final class MetricsFile {

    private MetricsFile() {}

    /** Base path the reporters write to (each appends {@code .<uuid>}): env
     *  {@code CLUSTERING_METRICS_FILE}, else system property {@code clustering.metrics.file},
     *  else a temp file. */
    public static String path() {
        String p = System.getenv("CLUSTERING_METRICS_FILE");
        if (p == null || p.isEmpty()) {
            p = System.getProperty("clustering.metrics.file",
                System.getProperty("java.io.tmpdir") + "/flink-metrics.txt");
        }
        return p;
    }

    /** True when the path was set explicitly (env or system property), i.e. NOT the
     *  node-local {@code java.io.tmpdir} fallback. On a real cluster the path must be an
     *  explicit shared-filesystem location, or the driver will only see its own node's
     *  files; this lets the caller fail fast in that case. */
    public static boolean isExplicitlyConfigured() {
        String env = System.getenv("CLUSTERING_METRICS_FILE");
        if (env != null && !env.isEmpty()) {
            return true;
        }
        return System.getProperty("clustering.metrics.file") != null;
    }

    /** Delete all reporter files for a clean run. */
    public static void reset() {
        for (Path f : reporterFiles()) {
            try {
                Files.deleteIfExists(f);
            } catch (IOException ignore) {
                // best-effort
            }
        }
    }

    /** All files written by the reporter instances: {@code <basename>.*} in the base dir. */
    private static List<Path> reporterFiles() {
        List<Path> out = new ArrayList<>();
        Path base = Paths.get(path());
        Path dir = base.getParent() == null ? Paths.get(".") : base.getParent();
        String prefix = base.getFileName().toString() + ".";
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, prefix + "*")) {
            for (Path p : ds) {
                if (!p.getFileName().toString().endsWith(".writing")) {
                    out.add(p);
                }
            }
        } catch (IOException ignore) {
            // dir missing -> no files
        }
        return out;
    }

    /** Parse + aggregate all reporter files into a Snapshot. No files -> all zeros. */
    public static BenchmarkListener.Snapshot read() {
        BenchmarkListener.Snapshot s = new BenchmarkListener.Snapshot();
        long srcOut = 0, sinkIn = 0, shuffleBytes = 0, cpuNs = 0, heap = 0, heapMax = 0, gc = 0;
        Set<String> taskIds = new HashSet<>();

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
                // Shuffle: count ONLY the Shuffle.Netty.Input scope (local + remote). The
                // operator-level numBytesInRemote is an alias of the same value -> matching
                // ".numBytesInRemote" loosely would double-count. local+remote = total shuffle
                // volume (matches Spark shuffleReadBytes; Flink counts all inter-task network).
                if (id.endsWith(".Shuffle.Netty.Input.numBytesInLocal")
                        || id.endsWith(".Shuffle.Netty.Input.numBytesInRemote")) {
                    shuffleBytes += v;
                } else if (id.endsWith(".numBytesOut") && id.contains("Source:")) {
                    srcOut += v;                   // input volume = what sources emit
                } else if (id.endsWith(".numBytesIn") && id.contains("Sink:")) {
                    sinkIn += v;                   // output volume = what reaches sinks
                } else if (id.endsWith(".numRecordsIn") && !id.contains("Shuffle.Netty")) {
                    taskIds.add(id);               // one per task instance
                } else if (tm && id.endsWith(".Status.JVM.CPU.Time")) {
                    cpuNs += v;
                } else if (tm && id.endsWith(".Status.JVM.Memory.Heap.Used")) {
                    heap += v;                      // SUM across TMs (cluster footprint)
                    heapMax = Math.max(heapMax, v); // MAX single TM
                } else if (tm && id.contains("GarbageCollector") && id.endsWith(".Time")) {
                    gc += v;  // sum across collectors (Young + Old) and TMs
                }
            }
        }

        s.inputBytes = srcOut;                 // Spark InputMetrics.bytesRead (source only)
        s.outputBytes = sinkIn;                // Spark OutputMetrics.bytesWritten (sink only)
        s.shuffleReadBytes = shuffleBytes;     // total shuffle volume (local+remote)
        s.shuffleWriteBytes = shuffleBytes;    // Flink has no read/write split -> same figure
        s.executorCpuTimeNs = cpuNs;           // process JVM CPU (see class doc; Spark differs)
        s.peakExecutorMemoryBytes = heapMax;   // MAX single TM heap
        s.totalExecutorMemoryBytes = heap;     // SUM of per-TM heap peaks (cluster footprint)
        s.jvmGcTimeMs = gc;
        s.taskCount = taskIds.size();
        // failedTaskCount, spill, stage*, executorRunTimeMs: no Flink analogue -> not set;
        // the driver leaves those RunResult fields null (N/A).
        return s;
    }
}
