package clustering.core;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.DataStreamUtils;
import org.apache.flink.streaming.api.operators.collect.ClientAndIterator;
import org.apache.flink.util.CloseableIterator;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Utility methods for collecting and consuming records from Flink DataStreams.
 */
public final class FlinkJobs {

    private FlinkJobs() {}

    /**
     * Drains the entire bounded stream into a list.
     */
    public static <T> List<T> collectAll(DataStream<T> stream, String jobName) {
        return collectUpTo(stream, jobName, -1);
    }

    /**
     * Drains the stream and returns the last emitted element (or null if empty).
     */
    public static <T> T last(DataStream<T> stream, String jobName) {
        List<T> elements = collectAll(stream, jobName);
        return elements.isEmpty() ? null : elements.get(elements.size() - 1);
    }

    /**
     * Iteratively consumes the stream as records arrive without loading the full result set
     * into driver memory. Safe for folding outputs larger than the driver's heap.
     *
     * @return the number of consumed records
     */
    public static <T> long consume(DataStream<T> stream, String jobName, Consumer<T> consumer) {
        try {
            ClientAndIterator<T> clientAndIterator = DataStreamUtils.collectWithClient(stream, jobName);
            long consumedCount = 0L;

            try (CloseableIterator<T> iterator = clientAndIterator.iterator) {
                while (iterator.hasNext()) {
                    consumer.accept(iterator.next());
                    consumedCount++;
                }
            }
            return consumedCount;
        } catch (Exception e) {
            throw new RuntimeException("Flink job '" + jobName + "' failed", e);
        }
    }

    /**
     * Collects up to the specified limit of records. If limit < 0, collects all.
     */
    public static <T> List<T> collectUpTo(DataStream<T> stream, String jobName, long limit) {
        try {
            ClientAndIterator<T> clientAndIterator = DataStreamUtils.collectWithClient(stream, jobName);
            List<T> collectedElements = new ArrayList<>();

            try (CloseableIterator<T> iterator = clientAndIterator.iterator) {
                while (iterator.hasNext() && (limit < 0 || collectedElements.size() < limit)) {
                    collectedElements.add(iterator.next());
                }
            }
            return collectedElements;
        } catch (Exception e) {
            throw new RuntimeException("Flink job '" + jobName + "' failed", e);
        }
    }
}
