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

/**
 * Materializes a {@link PointSource} once to shared Parquet storage.
 * Returns a new {@link PointSource} that reads the materialized copy,
 * allowing efficient data reuse across multiple Flink execution environments.
 */
public final class MaterializedPointSource implements PointSource {

    private static final Schema SCHEMA = SchemaBuilder.record("MaterializedPoint")
        .fields()
        .name("features").type().array().items().doubleType().noDefault()
        .name("weight").type().doubleType().noDefault()
        .endRecord();

    private final String storagePath;

    private MaterializedPointSource(String storagePath) {
        this.storagePath = storagePath;
    }

    @Override
    public DataStream<WeightedPoint> create(StreamExecutionEnvironment env) {
        FileSource<GenericRecord> fileSource = FileSource
            .forRecordStreamFormat(AvroParquetReaders.forGenericRecord(SCHEMA),
                new org.apache.flink.core.fs.Path(storagePath))
            .build();

        DataStream<GenericRecord> records = env.fromSource(
            fileSource, WatermarkStrategy.noWatermarks(), "materialized:" + storagePath,
            new GenericRecordAvroTypeInfo(SCHEMA));

        return records.map(new MapToWeightedPoint()).returns(WeightedPointTypeInfo.INSTANCE);
    }

    /**
     * Materializes the original source to Parquet storage and counts records via an accumulator.
     * Blocks execution until the file is fully committed.
     */
    public static MaterializationResult materialize(PointSource originalSource, EnvFactory envFactory, String targetPath) {
        StreamExecutionEnvironment env = envFactory.newEnv();

        DataStream<GenericRecord> records = originalSource.create(env)
            .map(new MapToGenericRecord())
            .returns(new GenericRecordAvroTypeInfo(SCHEMA));

        records.sinkTo(
            FileSink.forBulkFormat(new org.apache.flink.core.fs.Path(targetPath),
                    AvroParquetWriters.forGenericRecord(SCHEMA))
                .withBucketAssigner(new BasePathBucketAssigner<>())
                .build())
            .name("materialize-source-sink");

        try {
            JobExecutionResult jobResult = env.execute("materialize-source:" + targetPath);
            long totalRowCount = jobResult.getAccumulatorResult(MapToGenericRecord.ROW_COUNT_ACCUMULATOR);
            return new MaterializationResult(totalRowCount, new MaterializedPointSource(targetPath));
        } catch (Exception e) {
            throw new RuntimeException("Flink job 'materialize-source' failed", e);
        }
    }

    /**
     * Recursively deletes the materialized data at the given path.
     */
    public static void delete(String path) {
        try {
            org.apache.flink.core.fs.Path storagePath = new org.apache.flink.core.fs.Path(path);
            org.apache.flink.core.fs.FileSystem fileSystem = storagePath.getFileSystem();
            fileSystem.delete(storagePath, true);
        } catch (Exception e) {
            // Best-effort cleanup
        }
    }

    public static final class MaterializationResult {
        public final long rowCount;
        public final PointSource source;

        private MaterializationResult(long rowCount, PointSource source) {
            this.rowCount = rowCount;
            this.source = source;
        }
    }

    private static final class MapToGenericRecord extends RichMapFunction<WeightedPoint, GenericRecord> {
        static final String ROW_COUNT_ACCUMULATOR = "materializedRowCount";
        private transient LongCounter rowCounter;

        @Override
        public void open(Configuration parameters) {
            rowCounter = new LongCounter();
            getRuntimeContext().addAccumulator(ROW_COUNT_ACCUMULATOR, rowCounter);
        }

        @Override
        public GenericRecord map(WeightedPoint point) {
            rowCounter.add(1L);
            GenericRecord record = new GenericData.Record(SCHEMA);

            double[] features = point.features.values;
            List<Double> featureList = new ArrayList<>(features.length);
            for (double feature : features) {
                featureList.add(feature);
            }

            record.put("features", featureList);
            record.put("weight", point.weight);
            return record;
        }
    }

    private static final class MapToWeightedPoint implements MapFunction<GenericRecord, WeightedPoint> {
        @Override
        @SuppressWarnings("unchecked")
        public WeightedPoint map(GenericRecord record) {
            List<Double> featureList = (List<Double>) record.get("features");
            double[] featureArray = new double[featureList.size()];

            for (int i = 0; i < featureArray.length; i++) {
                featureArray[i] = featureList.get(i);
            }

            double weight = (Double) record.get("weight");
            return new WeightedPoint(new DenseVector(featureArray), weight);
        }
    }
}
