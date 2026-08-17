package clustering.core;

import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.io.Serializable;

/** (Re)creates the bounded point stream on a given execution environment.
 *
 *  Algorithms that run one Flink job per step (CLARA's sample and cost jobs) rebuild the
 *  source on a fresh env each time; the FLIP-176 ones build it once, inside the single
 *  iteration job. Matches {@code DataSource::load}.
 *
 *  <h3>Why DenseVector and not double[]</h3>
 *  {@link DenseVector} is the ecosystem's point type (Flink ML's own algorithms carry it), the
 *  counterpart of the {@code VectorUDT} column the Spark side uses. It is the type of the
 *  stream RECORD only: every operator unwraps {@code vector.values} — a field read, not a copy —
 *  before its hot loop, and caches/state hold plain {@code double[]}. So the distance kernels of
 *  both engines run over identical raw arrays, and the wrapper never enters an inner loop. */
@FunctionalInterface
public interface PointSource extends Serializable {
    DataStream<DenseVector> create(StreamExecutionEnvironment env);
}