package clustering.core;

import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.streaming.api.datastream.DataStream;

/**
 * Allocates Flink TaskManager's managed memory for an operator.
 * Ensures iterative algorithms cache data in memory rather than spilling to disk.
 */
public final class ManagedMemory {

    private static final long CACHE_WEIGHT_BYTES = 100L << 20; // 100 MB

    private ManagedMemory() {}

    /**
     * Grants the operator behind the stream access to the slot's managed memory for caching.
     */
    public static void allocateForPointCache(DataStream<?> stream) {
        DataStreamUtils.setManagedMemoryWeight(stream, CACHE_WEIGHT_BYTES);
    }
}
