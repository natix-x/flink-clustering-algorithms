package clustering.benchmark.datasource;

import clustering.benchmark.config.Params;
import clustering.core.WeightedPoint;
import clustering.core.WeightedPointTypeInfo;
import org.apache.avro.Schema;
import org.apache.avro.generic.GenericRecord;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.formats.avro.typeutils.GenericRecordAvroTypeInfo;
import org.apache.flink.formats.parquet.avro.AvroParquetReaders;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.avro.AvroSchemaConverter;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Reads benchmark datasets from Parquet files using an Avro record reader.
 * Infers schema from the Parquet footer, applies optional Bernoulli sampling,
 * and parses optional per-row weights.
 *
 * Note: The {@code numPartitions} parameter is reported but does not trigger
 * an explicit reshuffle, relying instead on Flink's native file source splitting.
 */
public final class ParquetDataSource implements DataSource {

    private final String path;
    private final String featureColumnName;
    private final Integer targetPartitionCount;
    private final String weightColumnName;
    private final Double sampleFraction;
    private final long sampleSeed;

    public ParquetDataSource(
            String path,
            String featureColumnName,
            Integer targetPartitionCount,
            String weightColumnName,
            Double sampleFraction,
            long sampleSeed
    ) {
        this.path = path;
        this.featureColumnName = featureColumnName;
        this.targetPartitionCount = targetPartitionCount;
        this.weightColumnName = weightColumnName;
        this.sampleFraction = sampleFraction;
        this.sampleSeed = sampleSeed;
    }

    @Override
    public String name() {
        return "parquet";
    }

    @Override
    public Map<String, String> metadata() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("format", "parquet");
        m.put("path", path);
        m.put("featureColumnName", featureColumnName);
        m.put("sampleFraction", sampleFraction == null ? "none" : sampleFraction.toString());
        m.put("seed", Long.toString(sampleSeed));
        m.put("numPartitions", targetPartitionCount == null ? "none" : targetPartitionCount.toString());
        m.put("weightColumn", weightColumnName == null ? "none" : weightColumnName);
        return m;
    }

    @Override
    public DataStream<WeightedPoint> load(StreamExecutionEnvironment env) {
        Schema schema = readSchema(path);
        Schema.Field field = schema.getField(featureColumnName);
        if (field == null) {
            throw new IllegalArgumentException(
                "ParquetDataSource: no column '" + featureColumnName + "' in " + path
                + " (columns: " + columnNames(schema) + ")"
            );
        }
        requireFeatureColumn(field);

        int weightPosition = -1;
        if (weightColumnName != null) {
            Schema.Field weightField = schema.getField(weightColumnName);
            if (weightField == null) {
                throw new IllegalArgumentException(
                    "ParquetDataSource: no weight column '" + weightColumnName + "' in " + path
                    + " (columns: " + columnNames(schema) + ")"
                );
            }
            Schema weightType = unwrapNullable(weightField.schema());
            if (!isNumeric(weightType.getType())) {
                throw new IllegalArgumentException(
                    "ParquetDataSource: weight column '" + weightColumnName + "' must be numeric, got "
                    + weightType
                );
            }
            weightPosition = weightField.pos();
        }

        FileSource<GenericRecord> source = FileSource
            .forRecordStreamFormat(AvroParquetReaders.forGenericRecord(schema), new org.apache.flink.core.fs.Path(path))
            .build();

        DataStream<GenericRecord> records = env.fromSource(
            source,
            WatermarkStrategy.noWatermarks(),
            "parquet:" + path,
            new GenericRecordAvroTypeInfo(schema)
        );

        if (sampleFraction != null && sampleFraction < 1.0) {
            records = records.filter(new SeededBernoulli(sampleFraction, sampleSeed));
        }

        boolean wrappedElements = unwrapListRecord(
            unwrapNullable(unwrapNullable(field.schema()).getElementType())
        ) != null;

        return records
            .map(new ExtractPoint(field.pos(), weightPosition, wrappedElements))
            .returns(WeightedPointTypeInfo.INSTANCE);
    }

    /**
     * Validates that the feature column is a numeric array.
     * Supports both flat arrays (array<double>) and Parquet's 3-level LIST encoding.
     */
    private static void requireFeatureColumn(Schema.Field field) {
        Schema type = unwrapNullable(field.schema());
        if (type.getType() != Schema.Type.ARRAY || !isNumericElement(type.getElementType())) {
            throw new IllegalArgumentException(
                "ParquetDataSource: unsupported feature-column type for '" + field.name()
                + "': " + type + " — expected an array of numbers, either flat (array<double> /"
                + " array<float>) or in Parquet's 3-level LIST encoding (array<record{element: double}>)"
            );
        }
    }

    private static boolean isNumericElement(Schema elementType) {
        Schema element = unwrapNullable(elementType);
        if (isNumeric(element.getType())) {
            return true;
        }
        Schema inner = unwrapListRecord(element);
        return inner != null && isNumeric(inner.getType());
    }

    /**
     * Returns the payload schema inside a 3-level LIST element record, or null if it is not a record.
     */
    private static Schema unwrapListRecord(Schema element) {
        if (element.getType() != Schema.Type.RECORD || element.getFields().size() != 1) {
            return null;
        }
        return unwrapNullable(element.getFields().get(0).schema());
    }

    private static boolean isNumeric(Schema.Type type) {
        return type == Schema.Type.DOUBLE || type == Schema.Type.FLOAT
            || type == Schema.Type.LONG || type == Schema.Type.INT;
    }

    /**
     * Extracts the non-null payload type from a Parquet optional column (Avro union with null).
     */
    private static Schema unwrapNullable(Schema schema) {
        if (schema.getType() != Schema.Type.UNION) {
            return schema;
        }
        for (Schema branch : schema.getTypes()) {
            if (branch.getType() != Schema.Type.NULL) {
                return branch;
            }
        }
        return schema;
    }

    private static List<String> columnNames(Schema schema) {
        List<String> names = new ArrayList<>();
        for (Schema.Field f : schema.getFields()) {
            names.add(f.name());
        }
        return names;
    }

    /**
     * Infers the dataset's Avro schema from the first Parquet part file's footer.
     * Prioritizes the "parquet.avro.schema" metadata key if available.
     */
    private static Schema readSchema(String path) {
        Configuration conf = new Configuration();
        try {
            Path root = new Path(path);
            FileSystem fs = root.getFileSystem(conf);
            Path part = fs.getFileStatus(root).isDirectory() ? firstPartFile(fs, root) : root;

            try (ParquetFileReader reader = ParquetFileReader.open(HadoopInputFile.fromPath(part, conf))) {
                org.apache.parquet.hadoop.metadata.FileMetaData meta = reader.getFooter().getFileMetaData();
                String stored = meta.getKeyValueMetaData().get("parquet.avro.schema");
                if (stored != null) {
                    return new Schema.Parser().parse(stored);
                }
                return new AvroSchemaConverter(conf).convert(meta.getSchema());
            }
        } catch (Exception e) {
            throw new RuntimeException("ParquetDataSource: cannot read the parquet schema at " + path, e);
        }
    }

    private static Path firstPartFile(FileSystem fs, Path root) throws java.io.IOException {
        List<FileStatus> parts = new ArrayList<>();
        for (FileStatus status : fs.listStatus(root)) {
            String fileName = status.getPath().getName();
            if (status.isFile() && !fileName.startsWith("_") && !fileName.startsWith(".")) {
                parts.add(status);
            }
        }
        if (parts.isEmpty()) {
            throw new java.io.IOException("no parquet part files under " + root);
        }
        parts.sort(Comparator.comparing(s -> s.getPath().getName()));
        return parts.get(0).getPath();
    }

    /**
     * Maps an Avro GenericRecord to a WeightedPoint.
     * Widens numeric elements to double and defaults missing weights to 1.0.
     */
    private static final class ExtractPoint implements MapFunction<GenericRecord, WeightedPoint> {

        private final int featurePosition;
        private final int weightPosition;
        private final boolean wrappedElements;

        ExtractPoint(int featurePosition, int weightPosition, boolean wrappedElements) {
            this.featurePosition = featurePosition;
            this.weightPosition = weightPosition;
            this.wrappedElements = wrappedElements;
        }

        @Override
        public WeightedPoint map(GenericRecord record) {
            Object raw = record.get(featurePosition);
            if (raw == null) {
                throw new IllegalStateException(
                    "ParquetDataSource: null feature vector in the column at position " + featurePosition
                );
            }

            @SuppressWarnings("unchecked")
            List<Object> values = (List<Object>) raw;
            double[] coords = new double[values.size()];

            if (wrappedElements) {
                for (int i = 0; i < coords.length; i++) {
                    coords[i] = ((Number) ((GenericRecord) values.get(i)).get(0)).doubleValue();
                }
            } else {
                for (int i = 0; i < coords.length; i++) {
                    coords[i] = ((Number) values.get(i)).doubleValue();
                }
            }

            double weight = 1.0;
            if (weightPosition >= 0) {
                Object rawWeight = record.get(weightPosition);
                if (rawWeight == null) {
                    throw new IllegalStateException(
                        "ParquetDataSource: null weight in the column at position " + weightPosition
                    );
                }
                weight = ((Number) rawWeight).doubleValue();
            }

            return new WeightedPoint(new DenseVector(coords), weight);
        }
    }

    /**
     * Seeded Bernoulli filter for row sampling.
     */
    private static final class SeededBernoulli extends org.apache.flink.api.common.functions.RichFilterFunction<GenericRecord> {

        private final double fraction;
        private final long seed;
        private transient Random rng;

        SeededBernoulli(double fraction, long seed) {
            this.fraction = fraction;
            this.seed = seed;
        }

        @Override
        public void open(org.apache.flink.configuration.Configuration parameters) {
            rng = new Random(seed + 31L * getRuntimeContext().getIndexOfThisSubtask());
        }

        @Override
        public boolean filter(GenericRecord value) {
            return rng.nextDouble() < fraction;
        }
    }

    public static Factory factory() {
        return new Factory() {
            @Override
            public String typeName() {
                return "parquet";
            }

            @Override
            public DataSource create(Map<String, Object> params) {
                Object numPartitions = params.get("numPartitions");
                Object weightColumn = params.get("weightColumn");
                Object fraction = params.get("sampleFraction");

                return new ParquetDataSource(
                    Params.stringParam(params, "path"),
                    Params.stringParam(params, "featureColumnName"),
                    numPartitions == null ? null : ((Number) numPartitions).intValue(),
                    weightColumn == null ? null : weightColumn.toString(),
                    fraction == null ? null : ((Number) fraction).doubleValue(),
                    Params.longParam(params, "seed", 42L)
                );
            }
        };
    }
}