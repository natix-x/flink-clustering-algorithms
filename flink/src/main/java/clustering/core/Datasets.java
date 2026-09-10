package clustering.core;

import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.List;
import java.util.OptionalLong;

/**
 * Driver-side helpers for executing Flink jobs and fetching results.
 */
public final class Datasets {

    private Datasets() {}

    /**
     * Collects the first 'limit' points deterministically (parallelism 1) to preserve source order.
     */
    public static List<WeightedPoint> takeFirst(PointSource source, EnvFactory envFactory, long limit) {
        StreamExecutionEnvironment env = envFactory.newEnv();
        env.setParallelism(1);
        return FlinkJobs.collectUpTo(source.create(env), "takeFirst", limit);
    }

    /**
     * Collects all points deterministically (parallelism 1) to preserve source order.
     */
    public static List<WeightedPoint> collectAll(PointSource source, EnvFactory envFactory) {
        StreamExecutionEnvironment env = envFactory.newEnv();
        env.setParallelism(1);
        return FlinkJobs.collectAll(source.create(env), "collectAll");
    }

    /**
     * Returns row count. Uses known size if available to avoid an extra Flink job execution.
     */
    public static long count(PointSource source, EnvFactory envFactory) {
        OptionalLong knownCount = source.knownRowCount();
        if (knownCount.isPresent()) {
            return knownCount.getAsLong();
        }

        StreamExecutionEnvironment env = envFactory.newEnv();
        DataStream<Long> counts = source.create(env)
            .map(point -> 1L).returns(Types.LONG)
            .keyBy(element -> 0)
            .reduce((a, b) -> a + b);

        Long totalCount = FlinkJobs.last(counts, "count");
        return totalCount == null ? 0L : totalCount;
    }
}
