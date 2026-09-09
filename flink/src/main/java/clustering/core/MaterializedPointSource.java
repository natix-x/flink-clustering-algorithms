package clustering.core;


import org.apache.avro.Schema;
import org.apache.avro.SchemaBuilder;
import org.apache.avro.generic.GenericData;
import org.apache.avro.generic.GenericRecord;
import org.apache.flink.api.common.JobExecutionResult;
import org.apache.flink.api.common.accumulators.LongCounter;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.sink.FileSink;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.formats.avro.typeutils.GenericRecordAvroTypeInfo;
import org.apache.flink.formats.parquet.avro.AvroParquetReaders;
import org.apache.flink.formats.parquet.avro.AvroParquetWriters;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.filesystem.bucketassigners.BasePathBucketAssigner;
import java.util.ArrayList;
import java.util.List;

/** Materialises a {@link PointSource} ONCE to shared storage (Parquet) and hands back a
 *  {@link PointSource} that reads the materialized copy instead of re-invoking the original
 *  connector.
 *
 *  <h3>Why this exists</h3>
 *  Every Flink action here runs on a fresh env ({@link EnvFactory}'s own doc), and many algorithms
 *  are BY DESIGN several sequential jobs on several fresh envs (CLARA's sample/cost jobs, PAMAE's
 *  seeding + refinement, dbscanpp's candidate/epsilon/graph phases — the P1/P2 patterns).
 *  {@code DataStream.cache()} is scoped to ONE env instance, so it cannot help any of those — only
 *  the FLIP-176 algorithms, which already read the source exactly once inside their single
 *  iteration job, would benefit. Materializing once to shared storage instead works with EVERY
 *  algorithm unchanged: whichever fresh env a later phase mints, reading this copy is a fast local
 *  scan instead of a real re-run of the original connector (synthetic generation, Parquet
 *  re-partitioning, a network-bound read). This is the Flink analogue of Spark's
 *  {@code persist(MEMORY_AND_DISK)+count()} — one real pass over the ORIGINAL source per run, so
 *  {@code fitDurationMs}/{@code evalDurationMs} stop secretly re-paying I/O that Spark's cached
 *  DataFrame doesn't pay.
 *
 *  <h3>Cost</h3>
 *  Unlike Spark's cache, this always pays a disk write during {@code load} (it cannot stay
 *  purely in-memory) — the honest trade for working across every algorithm's existing
 *  fresh-env-per-job architecture unchanged. */
public final class MaterializedPointSource implements PointSource {

    private static final Schema SCHEMA = SchemaBuilder.record("MaterializedPoint")
        .fields()
        .name("features").type().array().items().doubleType().noDefault()
        .name("weight").type().doubleType().noDefault()
        .endRecord();

    private final String path;

    private MaterializedPointSource(String path) {
        this.path = path;
    }

    @Override
    public DataStream<WeightedPoint> create(StreamExecutionEnvironment env) {
        FileSource<GenericRecord> fileSource = FileSource
            .forRecordStreamFormat(AvroParquetReaders.forGenericRecord(SCHEMA),
                new org.apache.flink.core.fs.Path(path))
            .build();
        DataStream<GenericRecord> records = env.fromSource(
            fileSource, WatermarkStrategy.noWatermarks(), "materialized:" + path,
            new GenericRecordAvroTypeInfo(SCHEMA));
        return records.map(new FromRecord()).returns(WeightedPointTypeInfo.INSTANCE);
    }

    /** Read the original source ONCE, writing it to {@code path} (Parquet, on shared storage —
     *  the caller's {@code outputDir}, so every TaskManager can see it) while counting rows via
     *  an accumulator, no separate counting pass. Blocks on {@link StreamExecutionEnvironment
     *  #execute} (not a collect-based job) so the file is fully committed — writer AND the
     *  BATCH-mode global commit — before this returns; a collect-sink job only guarantees the
     *  collect operator finished, not the file sink's commit phase. */
    public static Result materialize(PointSource original, EnvFactory envs, String path) {
        StreamExecutionEnvironment env = envs.newEnv();
        DataStream<GenericRecord> records = original.create(env)
            .map(new ToRecord()).returns(new GenericRecordAvroTypeInfo(SCHEMA));
        records.sinkTo(
            FileSink.forBulkFormat(new org.apache.flink.core.fs.Path(path),
                    AvroParquetWriters.forGenericRecord(SCHEMA))
                .withBucketAssigner(new BasePathBucketAssigner<>())
                .build())
            .name("materialize-source-sink");
        try {
            JobExecutionResult result = env.execute("materialize-source:" + path);
            long nRows = result.getAccumulatorResult(ToRecord.ROW_COUNT_ACCUMULATOR);
            return new Result(nRows, new MaterializedPointSource(path));
        } catch (Exception e) {
            throw new RuntimeException("Flink job 'materialize-source' failed", e);
        }
    }

    /** Recursively removes the materialized copy at {@code path}. Safe to call even if
     *  materialization never got far enough to create anything. */
    public static void delete(String path) {
        try {
            org.apache.flink.core.fs.Path p = new org.apache.flink.core.fs.Path(path);
            org.apache.flink.core.fs.FileSystem fs = p.getFileSystem();
            fs.delete(p, true);
        } catch (Exception e) {
            // Best-effort: a leftover temp dir is a disk-space nuisance, not a run failure.
        }
    }

    public static final class Result {
        public final long nRows;
        public final PointSource source;

        private Result(long nRows, PointSource source) {
            this.nRows = nRows;
            this.source = source;
        }
    }

    private static final class ToRecord extends RichMapFunction<WeightedPoint, GenericRecord> {
        static final String ROW_COUNT_ACCUMULATOR = "materializedRowCount";
        private transient LongCounter counter;

        @Override
        public void open(Configuration parameters) {
            counter = new LongCounter();
            getRuntimeContext().addAccumulator(ROW_COUNT_ACCUMULATOR, counter);
        }

        @Override
        public GenericRecord map(WeightedPoint p) {
            counter.add(1L);
            GenericRecord r = new GenericData.Record(SCHEMA);
            double[] values = p.features.values;
            List<Double> boxed = new ArrayList<>(values.length);
            for (double v : values) boxed.add(v);
            r.put("features", boxed);
            r.put("weight", p.weight);
            return r;
        }
    }

    private static final class FromRecord implements MapFunction<GenericRecord, WeightedPoint> {
        @Override
        @SuppressWarnings("unchecked")
        public WeightedPoint map(GenericRecord r) {
            List<Double> values = (List<Double>) r.get("features");
            double[] coords = new double[values.size()];
            for (int i = 0; i < coords.length; i++) {
                coords[i] = values.get(i);
            }
            double weight = (Double) r.get("weight");
            return new WeightedPoint(new DenseVector(coords), weight);
        }
    }
}
