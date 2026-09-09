package clustering.core;


import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.DataStreamUtils;
import org.apache.flink.streaming.api.operators.collect.ClientAndIterator;
import org.apache.flink.util.CloseableIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/** Thin wrappers around {@code DataStream} collection. Engine metrics are NOT gathered here
 *  — they are emitted by Flink's metric system to {@code clustering.metrics.FileMetricReporter}
 *  and read back by {@code clustering.benchmark.metrics.MetricsFile}, so no per-job result
 *  capture or REST polling is needed. */
public final class FlinkJobs {

    private FlinkJobs() {}

    /** Drain the full bounded stream into a list. */
    public static <T> List<T> collectAll(DataStream<T> stream, String jobName) {
        return collectUpTo(stream, jobName, -1);
    }

    /** Drain the full bounded stream, return the LAST element (or null) — for one-value
     *  reduces and FLIP-176 iterations whose final emitted record is the result. */
    public static <T> T last(DataStream<T> stream, String jobName) {
        List<T> all = collectAll(stream, jobName);
        return all.isEmpty() ? null : all.get(all.size() - 1);
    }

    /** Drain the full bounded stream, handing each record to {@code consumer} AS IT ARRIVES and
     *  never keeping more than one in the driver's hands.
     *
     *  This is what {@code collectAll} cannot do and Spark has no equivalent of: Spark's
     *  {@code collect} materialises the whole result on the driver, so an unbounded-output job
     *  (the ε-graph edge scan, whose output is O(edges) and quadratic on a dense graph) has to be
     *  cut into blocks whose size is guessed in advance. Flink's collect is a back-pressured
     *  ITERATOR, so the driver can fold a result set far larger than its heap in one job — bounded
     *  by construction rather than by a size estimate.
     *
     *  @return the number of records consumed */
    public static <T> long consume(DataStream<T> stream, String jobName, Consumer<T> consumer) {
        try {
            ClientAndIterator<T> cai = DataStreamUtils.collectWithClient(stream, jobName);
            long count = 0L;
            try (CloseableIterator<T> it = cai.iterator) {
                while (it.hasNext()) {
                    consumer.accept(it.next());
                    count++;
                }
            }
            return count;
        } catch (Exception e) {
            throw new RuntimeException("Flink job '" + jobName + "' failed", e);
        }
    }

    /** Collect up to {@code limit} records ({@code limit < 0} = unbounded), then close. */
    public static <T> List<T> collectUpTo(DataStream<T> stream, String jobName, long limit) {
        try {
            ClientAndIterator<T> cai = DataStreamUtils.collectWithClient(stream, jobName);
            List<T> out = new ArrayList<>();
            try (CloseableIterator<T> it = cai.iterator) {
                while (it.hasNext() && (limit < 0 || out.size() < limit)) {
                    out.add(it.next());
                }
            }
            return out;
        } catch (Exception e) {
            throw new RuntimeException("Flink job '" + jobName + "' failed", e);
        }
    }
}
