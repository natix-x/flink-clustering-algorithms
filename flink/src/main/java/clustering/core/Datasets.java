package clustering.core;


import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import java.util.List;

/** Small driver-side helpers that run a Flink job to bring data (or a summary)
 *  back to the driver. Each uses a fresh env from the factory and routes through
 *  {@link FlinkJobs} so the job's engine metrics land in {@link RunMetrics}. */
public final class Datasets {

    private Datasets() {}

    /** Collect the first {@code limit} points DETERMINISTICALLY: forces parallelism 1
     *  so the prefix is in source/index order, independent of the configured compute
     *  parallelism. Used for reproducible KMeans init and silhouette sampling. */
    public static List<WeightedPoint> collectHead(PointSource source, EnvFactory envs, long limit) {
        StreamExecutionEnvironment env = envs.newEnv();
        env.setParallelism(1);
        return FlinkJobs.collectUpTo(source.create(env), "collectHead", limit);
    }

    /** Collect ALL points to the driver DETERMINISTICALLY (parallelism 1, source/index
     *  order). Used by driver-local algorithms (PAM, FastPAM) whose O(n²) distance
     *  matrix only fits small datasets anyway, so parallelism-1 collection is cheap and
     *  makes results reproducible (medoid tie-breaking depends on point order). */
    public static List<WeightedPoint> collectAll(PointSource source, EnvFactory envs) {
        StreamExecutionEnvironment env = envs.newEnv();
        env.setParallelism(1);
        return FlinkJobs.collectAll(source.create(env), "collectAll");
    }

    /** Distributed row count — one Flink job, UNLESS the source already knows its size
     *  ({@link PointSource#knownRowCount()}), in which case this is free.
     *
     *  That shortcut is what keeps `clara` and `dbscanpp` from re-counting a dataset the `load`
     *  phase counted moments earlier: on Flink a count is a full job that re-reads the source,
     *  so the redundant one cost a whole extra pass over the data per run. */
    public static long count(PointSource source, EnvFactory envs) {
        java.util.OptionalLong known = source.knownRowCount();
        if (known.isPresent()) {
            return known.getAsLong();
        }
        StreamExecutionEnvironment env = envs.newEnv();
        DataStream<Long> total = source.create(env)
            .map(p -> 1L).returns(Types.LONG)
            .keyBy(x -> 0).reduce((a, b) -> a + b);
        Long n = FlinkJobs.last(total, "count");  // BATCH keyed reduce emits one final value
        return n == null ? 0L : n;
    }


}
