package clustering.core;

import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.streaming.api.datastream.DataStream;

/** Declares that an operator wants a slice of the TaskManager's MANAGED memory.
 *
 *  <p>Every iterative algorithm here caches its local points in a
 *  {@code org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache} — the Flink
 *  counterpart of Spark's {@code persist(MEMORY_AND_DISK)} on the input DataFrame. That class
 *  asks the operator's {@code StreamConfig} how much managed memory the slot granted it and,
 *  when the answer is zero, builds its {@code DataCacheWriter} with a NULL segment pool:
 *
 *  <pre>
 *    double fraction = config.getManagedMemoryFractionOperatorUseCaseOfSlot(OPERATOR, ...);
 *    if (fraction &gt; 0) {{ pool = new LazyMemorySegmentPool(...); }}   // else pool stays null
 *  </pre>
 *
 *  A null pool means the memory tier does not exist and EVERY record goes straight to a
 *  {@code FileSegmentWriter}, i.e. to {@code io.tmp.dirs} on disk. The fraction is non-zero only
 *  if the operator declared the use case at graph-build time, which nothing here did — so the
 *  point cache was spilling in full on every round while the TaskManager's managed memory sat
 *  allocated and idle. Against Spark, which caches the same points in executor heap, that made
 *  every Flink measurement partly a disk benchmark.
 *
 *  <p>The weight is relative between the managed-memory operators sharing a slot. Each of these
 *  jobs has exactly one, so any positive value hands it the whole pool; the constant below is
 *  in bytes because {@link DataStreamUtils#setManagedMemoryWeight} shifts it down by 20 to get
 *  the integer weight it declares. */
public final class ManagedMemory {

    /** Relative weight for the per-subtask point cache (bytes; >> 20 = the declared weight). */
    private static final long POINT_CACHE_WEIGHT_BYTES = 100L << 20;

    private ManagedMemory() {}

    /** Grant the operator behind {@code stream} the slot's managed memory for its point cache.
     *  Call it on the stream returned by the {@code transform(...)} that installs the operator
     *  holding the {@code ListStateWithCache}. */
    public static void forPointCache(DataStream<?> stream) {
        DataStreamUtils.setManagedMemoryWeight(stream, POINT_CACHE_WEIGHT_BYTES);
    }
}
