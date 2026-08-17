package clustering.benchmark.datasource;

import clustering.benchmark.config.Params;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/** 5-mode 3D mixture, same distribution as the Spark {@code SyntheticDataSource}
 *  (noise + two overlapping Gaussians + elongated mode + dense distant mode).
 *  Points are generated distributively from a sequence source; each index gets a
 *  deterministic per-point RNG so runs are reproducible regardless of parallelism. */
public final class SyntheticDataSource implements DataSource {

    private final long numPoints;
    private final int numPartitions;
    private final long seed;

    public SyntheticDataSource(long numPoints, int numPartitions, long seed) {
        this.numPoints = numPoints;
        this.numPartitions = numPartitions;
        this.seed = seed;
    }

    @Override
    public String name() {
        return "synthetic-5mix-3d";
    }

    @Override
    public Map<String, String> metadata() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("format", "synthetic");
        m.put("generator", name());
        m.put("numPoints", Long.toString(numPoints));
        m.put("numPartitions", Integer.toString(numPartitions));
        m.put("seed", Long.toString(seed));
        m.put("nFeatures", "3");
        return m;
    }

    @Override
    public DataStream<DenseVector> load(StreamExecutionEnvironment env) {
        // Parallelism is taken from the environment (set per job by the EnvFactory),
        // NOT pinned here — so a sample can be collected deterministically at
        // parallelism 1 (index order) while fit runs at full parallelism. Each point
        // is generated from its index, so the dataset is identical regardless of
        // parallelism.
        return env.fromSequence(0L, numPoints - 1)
            .map(new GeneratePoint(seed))
            .returns(DenseVectorTypeInfo.INSTANCE);
    }

    /** Maps a point index to a sample, seeding a per-point RNG deterministically. */
    private static final class GeneratePoint implements MapFunction<Long, DenseVector> {
        private final long seed;

        GeneratePoint(long seed) {
            this.seed = seed;
        }

        @Override
        public DenseVector map(Long index) {
            Random rng = new Random(seed + index * 0x9E3779B97F4A7C15L);
            return new DenseVector(samplePoint(rng));
        }
    }

    /** One sample from the 5-mode mixture. */
    private static double[] samplePoint(Random rng) {
        double r = rng.nextDouble();
        if (r < 0.05) {
            return new double[]{
                rng.nextDouble() * 2000 - 1000,
                rng.nextDouble() * 2000 - 1000,
                rng.nextDouble() * 2000 - 1000
            };
        } else if (r < 0.30) {
            return new double[]{rng.nextGaussian() * 5, rng.nextGaussian() * 5, rng.nextGaussian() * 5};
        } else if (r < 0.55) {
            return new double[]{
                rng.nextGaussian() * 5 + 5, rng.nextGaussian() * 5 + 5, rng.nextGaussian() * 5 + 5
            };
        } else if (r < 0.70) {
            return new double[]{
                rng.nextGaussian() * 40 + 100,
                rng.nextGaussian() * 2 + 50,
                rng.nextGaussian() * 2 + 50
            };
        } else {
            return new double[]{
                rng.nextGaussian() + 200,
                rng.nextGaussian() + 200,
                rng.nextGaussian() + 200
            };
        }
    }

    public static Factory factory() {
        return new Factory() {
            @Override public String typeName() { return "synthetic"; }
            @Override public DataSource create(Map<String, Object> params) {
                return new SyntheticDataSource(
                    Params.longParam(params, "numPoints"),
                    Params.intParam(params, "numPartitions", 8),
                    Params.longParam(params, "seed", 42L));
            }
        };
    }
}
