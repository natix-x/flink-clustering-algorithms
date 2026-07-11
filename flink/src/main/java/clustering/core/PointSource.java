package clustering.core;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.io.Serializable;

/** (Re)creates the bounded point stream on a given execution environment.
 *
 *  Algorithms that run one Flink job per iteration (impl=jobiter) rebuild the
 *  source on a fresh env each round; iteration-based ones (impl=mliter) build it
 *  once. Matches {@code DataSource::load}. */
@FunctionalInterface
public interface PointSource extends Serializable {
    DataStream<double[]> create(StreamExecutionEnvironment env);
}