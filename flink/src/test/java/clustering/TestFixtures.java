package clustering;

import clustering.core.EnvFactory;
import clustering.core.Model;
import clustering.core.PointSource;
import clustering.core.Points;
import org.apache.flink.ml.linalg.DenseVector;
import clustering.distance.DistanceMetric;
import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Shared test scaffolding: a local {@link EnvFactory}, in-memory {@link PointSource}s and the
 *  label-comparison helpers the algorithm specs assert on. */
public final class TestFixtures {

    private TestFixtures() {}

    /** Fresh BATCH MiniCluster env per Flink action, exactly as the local run profile does. */
    public static EnvFactory localEnvs(int parallelism) {
        return () -> {
            StreamExecutionEnvironment env = StreamExecutionEnvironment.createLocalEnvironment(parallelism);
            env.setRuntimeMode(RuntimeExecutionMode.BATCH);
            env.setParallelism(parallelism);
            return env;
        };
    }

    public static PointSource source(List<double[]> points) {
        return env -> env.fromCollection(Points.wrapAll(points), DenseVectorTypeInfo.INSTANCE);
    }

    /** {@code count} x {@code count} grid of spacing {@code step}, anchored at (ox, oy). */
    public static List<double[]> grid(double ox, double oy, int count, double step) {
        List<double[]> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < count; j++) {
                out.add(new double[] {ox + i * step, oy + j * step});
            }
        }
        return out;
    }

    /** The partition a label array induces, as point-index groups; noise ({@code -1}) is
     *  returned separately. Cluster IDS are never compared — only the grouping, since ids are
     *  arbitrary between two implementations. */
    public static Set<Set<Integer>> clustersOf(int[] labels) {
        Map<Integer, Set<Integer>> byLabel = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] != -1) {
                byLabel.computeIfAbsent(labels[i], key -> new HashSet<>()).add(i);
            }
        }
        return new HashSet<>(byLabel.values());
    }

    public static Set<Integer> noiseOf(int[] labels) {
        Set<Integer> noise = new HashSet<>();
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] == -1) {
                noise.add(i);
            }
        }
        return noise;
    }

    /** phi(C, X) = sum over points of d(x, nearest prototype)^2 — for comparing two fits. */
    public static double totalError(Model model, List<double[]> points, DenseVector[] prototypes,
                                    DistanceMetric distance) {
        double total = 0.0;
        for (double[] p : points) {
            double d = distance.compute(p, prototypes[model.predict(Points.wrap(p))].values);
            total += d * d;
        }
        return total;
    }
}
