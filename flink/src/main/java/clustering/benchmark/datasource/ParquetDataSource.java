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

/** Reads the benchmark's real datasets. Java mirror of the Spark {@code ParquetDataSource}, same
 *  config params and same semantics.
 *
 *  <h3>Why the Avro record reader</h3>
 *  The harness writes features as an engine-neutral {@code array<double>} / {@code array<float>}
 *  column — never a Spark ML {@code VectorUDT} — exactly so this class can exist. Flink's
 *  vectorised {@code ParquetColumnarRowInputFormat} only handles primitive columns in 1.17, and a
 *  feature vector is a repeated column, so the record-based Avro reader is the reader that can read
 *  our files at all. Element type is honoured as written: float embeddings are widened to double
 *  once, at the read boundary, so nothing downstream branches on it.
 *
 *  <h3>Schema inference</h3>
 *  Spark infers the schema; the Avro reader needs it up front, so it is read from a part file's
 *  parquet FOOTER on the driver while the job graph is built (one small read, no data scan) and
 *  converted with {@link AvroSchemaConverter}. That keeps the config to the same two required
 *  params Spark takes — {@code path} and {@code featureColumnName} — instead of making every run
 *  restate the dimensionality.
 *
 *  <h3>Sampling</h3>
 *  {@code sampleFraction} is a seeded Bernoulli filter applied right after the read, so rows are
 *  dropped before anything else touches them — the counterpart of Spark sampling before the
 *  projection so both stay pushable into the scan. Each subtask seeds from {@code seed} and its
 *  subtask id, but the drawn SUBSET is not fixed: split-to-subtask assignment varies between runs,
 *  so a second run of the same config reads a different sample of the same size. Spark's
 *  {@code sample} is partition-dependent for the same reason, so the two engines are on equal
 *  footing here — which is the point. The guarantee is the distribution and the expected size, not
 *  the identity of the rows.
 *
 *  <h3>Weights</h3>
 *  {@code weightColumn} names an optional per-row weight — how many points the row stands for.
 *  Absent means 1.0, so a config without it keeps its meaning exactly. The value is widened to
 *  {@code double} here, at the same boundary as the features, so the rest of the engine never asks
 *  whether the input was weighted.
 *
 *  <h3>{@code numPartitions}</h3>
 *  Accepted and reported, but it does NOT reshuffle. On Spark it coalesces/repartitions the loaded
 *  frame because the read splits by file size and can leave cores idle. Flink's file source assigns
 *  splits to subtasks dynamically, so the run's parallelism — set per job by the {@code EnvFactory}
 *  — already governs how the work spreads, and forcing a rebalance here would add a shuffle Spark
 *  does not pay either. Recorded as a deliberate asymmetry rather than silently ignored: a config
 *  that sets it gets the same row set on both engines, distributed by each engine's own rule. */
public final class ParquetDataSource implements DataSource {

    private final String path;
    private final String featureColumnName;
    private final Integer targetPartitionCount;
    private final String weightColumnName;
    private final Double sampleFraction;
    private final long sampleSeed;

    public ParquetDataSource(String path, String featureColumnName, Integer targetPartitionCount,
                             String weightColumnName, Double sampleFraction, long sampleSeed) {
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
        m.put("numPartitions",
            targetPartitionCount == null ? "none" : targetPartitionCount.toString());
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
                + " (columns: " + columnNames(schema) + ")");
        }
        requireFeatureColumn(field);

        int weightPosition = -1;
        if (weightColumnName != null) {
            Schema.Field weightField = schema.getField(weightColumnName);
            if (weightField == null) {
                throw new IllegalArgumentException(
                    "ParquetDataSource: no weight column '" + weightColumnName + "' in " + path
                    + " (columns: " + columnNames(schema) + ")");
            }
            Schema weightType = unwrapNullable(weightField.schema());
            if (!isNumeric(weightType.getType())) {
                throw new IllegalArgumentException(
                    "ParquetDataSource: weight column '" + weightColumnName + "' must be numeric, got "
                    + weightType);
            }
            weightPosition = weightField.pos();
        }

        FileSource<GenericRecord> source = FileSource
            .forRecordStreamFormat(AvroParquetReaders.forGenericRecord(schema),
                new org.apache.flink.core.fs.Path(path))
            .build();

        DataStream<GenericRecord> records = env.fromSource(
            source, WatermarkStrategy.noWatermarks(), "parquet:" + path,
            // Avro-aware type info: GenericRecord carries no schema in its class, so the Kryo
            // fallback cannot round-trip it.
            new GenericRecordAvroTypeInfo(schema));

        if (sampleFraction != null && sampleFraction < 1.0) {
            records = records.filter(new SeededBernoulli(sampleFraction, sampleSeed));
        }

        // The field POSITION, resolved once here, not the name: `GenericRecord.get(String)` is a
        // schema hash lookup per record, and this map runs on every row of a 434-million-row set.
        // It also keeps the closure free of a String field, which Flink 1.17's ClosureCleaner
        // cannot reflect into under JDK 17.
        // Whether elements arrive wrapped is a property of the FILE (see readSchema), decided
        // once here and never per row — this map runs on every row of a 434-million-row set.
        boolean wrappedElements = unwrapListRecord(
            unwrapNullable(unwrapNullable(field.schema()).getElementType())) != null;

        return records
            .map(new ExtractPoint(field.pos(), weightPosition, wrappedElements))
            .returns(WeightedPointTypeInfo.INSTANCE);
    }

    /** Rejects a feature column that is not a numeric list, with the same intent as the Spark
     *  side's "unsupported feature-column type" guard — better a config error than a run that
     *  produces nonsense vectors.
     *
     *  <p>Accepts BOTH shapes a Parquet list can arrive in, which is the difference between
     *  reading the project's real datasets and not (found on Ares, 5.09.2026):
     *  <ul>
     *    <li>{@code array<double>} — the flat, 2-level encoding, what {@code AvroParquetWriter}
     *        produces and therefore all this class was ever tested against;</li>
     *    <li>{@code array<record{element: double}>} — the standard 3-level LIST encoding of the
     *        Parquet spec, which is what Spark (and so the harness' preprocessing jobs) actually
     *        writes for Gaia, NYC, Cohere and the rest.</li>
     *  </ul>
     *  Spark never has to choose: its Parquet reader normalises both into a plain
     *  {@code ArrayType} before any user code sees the schema. The Avro record reader used here
     *  surfaces the intermediate record instead, so the unwrapping has to happen explicitly —
     *  same end result, one layer lower. */
    private static void requireFeatureColumn(Schema.Field field) {
        Schema type = unwrapNullable(field.schema());
        if (type.getType() != Schema.Type.ARRAY || !isNumericElement(type.getElementType())) {
            throw new IllegalArgumentException(
                "ParquetDataSource: unsupported feature-column type for '" + field.name()
                + "': " + type + " — expected an array of numbers, either flat (array<double> /"
                + " array<float>) or in Parquet's 3-level LIST encoding"
                + " (array<record{element: double}>)");
        }
    }

    /** True for a list element that resolves to a number, through the 3-level LIST wrapper if
     *  there is one. */
    private static boolean isNumericElement(Schema elementType) {
        Schema element = unwrapNullable(elementType);
        if (isNumeric(element.getType())) {
            return true;
        }
        Schema inner = unwrapListRecord(element);
        return inner != null && isNumeric(inner.getType());
    }

    /** The payload schema inside a 3-level LIST element record, or {@code null} if this element is
     *  not such a record. The wrapper is a record with exactly ONE field (conventionally
     *  {@code element}, {@code item} or {@code array} depending on the writer), so the field's
     *  NAME is deliberately not matched — only its arity. */
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

    /** Parquet optional columns become an Avro union with null; the payload is the other branch. */
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

    /** The dataset's Avro schema, from a part file's parquet footer.
     *
     *  A path may be a single file or a directory of {@code part-*.parquet} files; the first part
     *  file in NAME order is read, so the inferred schema does not depend on directory listing
     *  order. Files parquet ignores in a normal read ({@code _SUCCESS}, {@code .crc}, hidden
     *  entries) are skipped here too. */
    private static Schema readSchema(String path) {
        Configuration conf = new Configuration();
        try {
            Path root = new Path(path);
            FileSystem fs = root.getFileSystem(conf);
            Path part = fs.getFileStatus(root).isDirectory() ? firstPartFile(fs, root) : root;
            try (ParquetFileReader reader =
                     ParquetFileReader.open(HadoopInputFile.fromPath(part, conf))) {
                org.apache.parquet.hadoop.metadata.FileMetaData meta =
                    reader.getFooter().getFileMetaData();
                // EXACTLY what parquet-avro will use, in its own order of preference: the Avro
                // schema stored in the file's key-value metadata if the writer left one, and only
                // otherwise the Parquet schema converted. Getting this wrong is not academic —
                // the two disagree on Parquet's 3-level LIST encoding, and the disagreement
                // decides whether a list element arrives as a Double or as a one-field
                // GenericRecord. A file written by AvroParquetWriter carries the key (so a flat
                // array<double> comes back flat); Gaia and every other dataset here is written by
                // SPARK, which stores no such key, so the same physical layout comes back wrapped.
                // Reading the converted schema unconditionally therefore worked in the tests and
                // failed on every real file ("UnresolvedUnionException: Not in union
                // [null,double]: {element: ...}").
                String stored = meta.getKeyValueMetaData().get("parquet.avro.schema");
                if (stored != null) {
                    return new Schema.Parser().parse(stored);
                }
                return new AvroSchemaConverter(conf).convert(meta.getSchema());
            }
        } catch (Exception e) {
            throw new RuntimeException(
                "ParquetDataSource: cannot read the parquet schema at " + path, e);
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

    /** One record -> its {@link WeightedPoint}. Widens whatever numeric element type the file uses
     *  to {@code double} ONCE, here at the boundary, so no downstream loop branches on it.
     *
     *  An absent weight column means weight 1.0 — the unweighted run is the unit-weight weighted
     *  run, which is why nothing downstream has to ask whether the input was weighted. */
    private static final class ExtractPoint implements MapFunction<GenericRecord, WeightedPoint> {

        private final int featurePosition;
        /** {@code -1} when the config named no weight column. */
        private final int weightPosition;
        /** True when this file's elements arrive as one-field records (Parquet's 3-level LIST
         *  encoding, as written by Spark) rather than as bare numbers. */
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
                    "ParquetDataSource: null feature vector in the column at position "
                    + featurePosition);
            }
            @SuppressWarnings("unchecked")
            List<Object> values = (List<Object>) raw;
            double[] coords = new double[values.size()];
            // Branch once per ROW, not per coordinate: at 1024 dims the inner loop runs 1024
            // times and the answer is the same every time.
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
                        "ParquetDataSource: null weight in the column at position " + weightPosition);
                }
                weight = ((Number) rawWeight).doubleValue();
            }
            return new WeightedPoint(new DenseVector(coords), weight);
        }
    }

    /** Seeded Bernoulli row filter; see the class javadoc on reproducibility. */
    private static final class SeededBernoulli
            extends org.apache.flink.api.common.functions.RichFilterFunction<GenericRecord> {

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
                    Params.longParam(params, "seed", 42L));
            }
        };
    }
}
