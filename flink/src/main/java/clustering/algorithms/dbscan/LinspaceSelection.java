package clustering.algorithms.dbscan;

import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.FlinkJobs;
import clustering.core.PointSource;
import clustering.core.Points;
import clustering.distance.DistanceMetric;
import org.apache.flink.api.common.functions.RichFilterFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.List;

/** Strided selection over the dataset's own order — no RNG, one pass, the cheapest of the three
 *  strategies. Honest weakness: on data ordered by position (Gaia by sky region, TLC by time) a
 *  stride can systematically miss regions.
 *
 *  Not a GLOBAL linspace, exactly like the Spark version: the row counter restarts in every
 *  subtask, so the stride has a phase jump at each partition boundary and the kept count
 *  deviates by O(parallelism) rather than by one row (a subtask narrower than the stride yields
 *  0 or 1 rows). A true global stride would need a global index, i.e. a shuffle. Plot the m
 *  that {@link DBSCANpp} logs, never the nominal one.
 *
 *  Two steps, not one modulo: no stride lands on exactly m rows, so the filter keeps AT LEAST
 *  m and the exact count comes from a linspace over the collected array. */
public final class LinspaceSelection implements CandidateSelectionStrategy {

    public static final LinspaceSelection INSTANCE = new LinspaceSelection();

    private LinspaceSelection() {}

    @Override
    public String strategyName() {
        return "linspace";
    }

    @Override
    public double[][] selectCandidates(PointSource source, EnvFactory envs, long totalRowCount,
                                       int targetCandidateCount, long seed, DistanceMetric distanceMetric) {
        if (targetCandidateCount >= totalRowCount) {
            return Points.toArray(Datasets.collectAll(source, envs));
        }
        StreamExecutionEnvironment env = envs.newEnv();
        DataStream<DenseVector> strided;
        if (2L * targetCandidateCount <= totalRowCount) {
            // Keep every stride-th row: >= m, < 2m kept.
            long stride = totalRowCount / targetCandidateCount;
            strided = source.create(env).filter(new StrideFilter(stride, true));
        } else {
            // Denser than every second row: DROP every stride-th row instead, else the stride
            // collapses to 1 and spreads nothing. >= m, <= n < 2m kept.
            long stride = (long) Math.ceil((double) totalRowCount / (totalRowCount - targetCandidateCount));
            strided = source.create(env).filter(new StrideFilter(stride, false));
        }
        List<double[]> rows = Points.toList(FlinkJobs.collectAll(strided, "dbscanpp-linspace-candidates"));
        if (rows.size() <= targetCandidateCount) {
            return rows.toArray(new double[0][]);
        }
        double[][] out = new double[targetCandidateCount][];
        for (int i = 0; i < targetCandidateCount; i++) {
            out[i] = rows.get((int) ((long) i * rows.size() / targetCandidateCount));
        }
        return out;
    }

    /** Per-subtask row stride: keeps (or drops) every {@code stride}-th row seen by this
     *  subtask. Stateless across runs, so it needs no seed. */
    static final class StrideFilter extends RichFilterFunction<DenseVector> {
        private final long stride;
        private final boolean keepMultiples;
        private transient long index;

        StrideFilter(long stride, boolean keepMultiples) {
            this.stride = Math.max(1L, stride);
            this.keepMultiples = keepMultiples;
        }

        @Override
        public void open(Configuration parameters) {
            index = 0L;
        }

        @Override
        public boolean filter(DenseVector value) {
            boolean isMultiple = index % stride == 0L;
            index++;
            return keepMultiples == isMultiple;
        }
    }
}
