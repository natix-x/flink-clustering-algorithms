package clustering.core;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.io.Serializable;
import java.util.OptionalLong;

/**
 * Creates a bounded stream of weighted points on a given execution environment.
 */
@FunctionalInterface
public interface PointSource extends Serializable {

    DataStream<WeightedPoint> create(StreamExecutionEnvironment env);

    /**
     * Returns the pre-computed row count if available.
     * Prevents executing redundant and expensive Flink counting jobs.
     */
    default OptionalLong knownRowCount() {
        return OptionalLong.empty();
    }

    /**
     * Wraps an existing source to explicitly provide its known row count.
     */
    static PointSource withKnownRowCount(PointSource originalSource, long rowCount) {
        return new PointSource() {
            @Override
            public DataStream<WeightedPoint> create(StreamExecutionEnvironment env) {
                return originalSource.create(env);
            }

            @Override
            public OptionalLong knownRowCount() {
                return OptionalLong.of(rowCount);
            }
        };
    }
}
