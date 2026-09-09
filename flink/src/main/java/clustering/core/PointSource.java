package clustering.core;


import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import java.io.Serializable;

/** (Re)creates the bounded point stream on a given execution environment.
 *
 *  Algorithms that run one Flink job per step (CLARA's sample and cost jobs) rebuild the
 *  source on a fresh env each time; the FLIP-176 ones build it once, inside the single
 *  iteration job. Matches {@code DataSource::load}.
 *
 *  <h3>Why WeightedPoint and not DenseVector</h3>
 *  The record carries the point AND how many points it stands for, because every Σ-over-points in
 *  the repo is weighted — the counterpart of the Spark side's {@code [features, weight]} row. An
 *  absent weight is 1.0, so an unweighted run is the unit-weight weighted run and nothing branches
 *  on "is this weighted".
 *
 *  {@link WeightedPoint#features} is still a {@link org.apache.flink.ml.linalg.DenseVector} — the
 *  ecosystem's point type, the counterpart of Spark's {@code VectorUDT}, and the type the fitted
 *  models carry. Both wrappers are RECORD-level only: every operator reads {@code features.values}
 *  (a field read, not a copy) before its hot loop, and caches/state hold plain {@code double[]}
 *  where they can, so the distance kernels of both engines run over identical raw arrays. */
@FunctionalInterface
public interface PointSource extends Serializable {
    DataStream<WeightedPoint> create(StreamExecutionEnvironment env);

    /** How many rows this source yields, when that is already known — so {@link Datasets#count}
     *  can answer without submitting a job.
     *
     *  <p>It is known for exactly one source: the one the benchmark job hands to the algorithms,
     *  because the `load` phase has just counted it. Without this, algorithms that need n
     *  ({@code clara}, {@code dbscanpp}) count it AGAIN, and on Flink a count is a full job that
     *  re-reads the source from storage — the row count of Cohere's 196 GB, paid twice per run to
     *  learn a number the job already had. Spark pays the same redundancy far more cheaply, over
     *  its persisted cache.
     *
     *  <p>Empty by default, and that default is the safety property: a DERIVED source (a filter, a
     *  sample) is a different {@code PointSource} instance and therefore reports nothing, so a
     *  count can never be inherited by a stream that does not have those rows. Only
     *  {@link #withKnownRowCount} attaches one, and only to the very source it wraps. */
    default java.util.OptionalLong knownRowCount() {
        return java.util.OptionalLong.empty();
    }

    /** The same source, carrying its already-known row count. */
    static PointSource withKnownRowCount(PointSource source, long rowCount) {
        return new PointSource() {
            @Override
            public DataStream<WeightedPoint> create(StreamExecutionEnvironment env) {
                return source.create(env);
            }

            @Override
            public java.util.OptionalLong knownRowCount() {
                return java.util.OptionalLong.of(rowCount);
            }
        };
    }
}
