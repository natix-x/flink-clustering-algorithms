package clustering.benchmark.datasource;

import clustering.TestFixtures;
import clustering.core.Datasets;
import clustering.core.EnvFactory;
import clustering.core.PointSource;
import org.apache.avro.Schema;
import org.apache.avro.SchemaBuilder;
import org.apache.avro.generic.GenericData;
import org.apache.avro.generic.GenericRecord;
import clustering.core.WeightedPoint;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.avro.AvroParquetWriter;
import org.apache.parquet.hadoop.ParquetWriter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Reads a REAL parquet file, written here in the shape the harness writes: a single
 *  {@code array<double>} (or {@code array<float>}) feature column, no Spark ML types.
 *
 *  Worth testing against a real file rather than a mock, because the two things most likely to
 *  break are exactly the two that a mock cannot exercise: the footer-based schema inference and
 *  the float-vs-double element type. */
class ParquetDataSourceSpec {

    private static final int PARALLELISM = 2;

    private static Schema featureSchema(Schema.Type elementType) {
        Schema element = Schema.create(elementType);
        return SchemaBuilder.record("Row").fields()
            .name("features").type(Schema.createArray(element)).noDefault()
            .endRecord();
    }

    /** Writes {@code rows} as one parquet part file under {@code dir}, elements typed as asked. */
    private static void writeParquet(java.nio.file.Path dir, List<double[]> rows,
                                     Schema.Type elementType) throws Exception {
        Schema schema = featureSchema(elementType);
        Path file = new Path(dir.resolve("part-00000.parquet").toString());
        try (ParquetWriter<GenericRecord> writer =
                 AvroParquetWriter.<GenericRecord>builder(file).withSchema(schema).build()) {
            for (double[] row : rows) {
                List<Object> values = new ArrayList<>(row.length);
                for (double v : row) {
                    values.add(elementType == Schema.Type.FLOAT ? (float) v : v);
                }
                GenericRecord record = new GenericData.Record(schema);
                record.put("features", values);
                writer.write(record);
            }
        }
        // A real Spark/harness output directory carries these; the reader must ignore them.
        Files.createFile(dir.resolve("_SUCCESS"));
    }

    private static DataSource source(java.nio.file.Path dir, Map<String, Object> extra) {
        Map<String, Object> params = new HashMap<>(extra);
        params.put("path", dir.toString());
        params.put("featureColumnName", "features");
        return ParquetDataSource.factory().create(params);
    }

    private static List<WeightedPoint> readAll(DataSource dataSource) {
        EnvFactory envs = TestFixtures.localEnvs(PARALLELISM);
        PointSource points = dataSource::load;
        return Datasets.collectAll(points, envs);
    }

    private static List<double[]> threeByFour() {
        List<double[]> rows = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            rows.add(new double[] {i, i + 0.5, -i, i * 2.0});
        }
        return rows;
    }

    @Test
    void readsADoubleArrayFeatureColumn(@TempDir java.nio.file.Path dir) throws Exception {
        List<double[]> rows = threeByFour();
        writeParquet(dir, rows, Schema.Type.DOUBLE);

        List<WeightedPoint> read = readAll(source(dir, Map.of()));

        assertEquals(rows.size(), read.size());
        assertEquals(4, read.get(0).size(), "dimensionality comes from the file, not the config");
        // Row ORDER is not part of the contract (splits are assigned dynamically), so compare sets.
        Set<String> expected = new HashSet<>();
        for (double[] row : rows) {
            expected.add(java.util.Arrays.toString(row));
        }
        for (WeightedPoint v : read) {
            assertTrue(expected.contains(java.util.Arrays.toString(v.values())),
                "unexpected row " + java.util.Arrays.toString(v.values()));
        }
    }

    /** Parquet's 3-LEVEL LIST encoding — {@code array<record{element: double}>} — which is what
     *  Spark, and therefore every preprocessing job in this project, actually writes.
     *
     *  <p>This is the case that was missing, and its absence was not harmless: the reader rejected
     *  the real Gaia files outright ("unsupported feature-column type"), so every Flink run of the
     *  parquet matrices on Ares failed at the source (5.09.2026). The old tests all passed because
     *  they wrote their fixtures with {@code AvroParquetWriter} straight from a flat
     *  {@code array<double>} schema — a test verifying the shape it had itself produced, which is
     *  the one shape production never sends.
     *
     *  <p>The naming is {@code list}/{@code element} on purpose, and it is not cosmetic: a Parquet
     *  repeated group holding exactly ONE field is ambiguous between the 3-level encoding and the
     *  legacy 2-level one, and parquet-avro disambiguates it by exactly these names. Writing the
     *  fixture with any other field name ({@code item}, say) makes the reader take the legacy path
     *  and fail with "optional double item is not a group" — which is how this test was first
     *  written, and the failure that proved the fixture, not the reader, was wrong. The reader's
     *  own unwrapping still keys off the record's ARITY rather than the name, so a file from a
     *  writer that names things differently is read correctly as long as Parquet agrees it is a
     *  3-level list. */
    @Test
    void readsTheThreeLevelListEncodingSparkWrites(@TempDir java.nio.file.Path dir) throws Exception {
        // A FLAT avro schema plus `write-old-list-structure=false` is what produces the CANONICAL
        // 3-level encoding — `repeated group list { optional double element; }` — i.e. bit-for-bit
        // the structure Spark writes. Left at its default, AvroParquetWriter emits the legacy
        // `repeated group array {...}` instead, which parquet-avro then cannot read back at all
        // ("optional double element is not a group"); an earlier version of this test wrote that
        // and blamed the reader.
        Schema schema = SchemaBuilder.record("Row").fields()
            .name("features").type(Schema.createArray(Schema.create(Schema.Type.DOUBLE))).noDefault()
            .endRecord();

        List<double[]> rows = threeByFour();
        Path file = new Path(dir.resolve("part-00000.parquet").toString());
        try (ParquetWriter<GenericRecord> writer =
                 AvroParquetWriter.<GenericRecord>builder(file)
                     .withSchema(schema)
                     .config("parquet.avro.write-old-list-structure", "false")
                     .build()) {
            for (double[] row : rows) {
                List<Object> values = new ArrayList<>(row.length);
                for (double v : row) {
                    values.add(v);
                }
                GenericRecord record = new GenericData.Record(schema);
                record.put("features", values);
                writer.write(record);
            }
        }
        Files.createFile(dir.resolve("_SUCCESS"));

        List<WeightedPoint> read = readAll(source(dir, Map.of()));

        assertEquals(rows.size(), read.size());
        assertEquals(4, read.get(0).size(), "dimensionality comes from the file, not the config");
        Set<String> expected = new HashSet<>();
        for (double[] row : rows) {
            expected.add(java.util.Arrays.toString(row));
        }
        for (WeightedPoint v : read) {
            assertTrue(expected.contains(java.util.Arrays.toString(v.values())),
                "unexpected row " + java.util.Arrays.toString(v.values())
                    + " — the 3-level wrapper was not unwrapped correctly");
        }
    }

    /** The embedding datasets are {@code array<float>}; the widening must happen at the read
     *  boundary so every downstream loop still sees a plain {@code double[]}. */
    @Test
    void readsAFloatArrayFeatureColumnAsDoubles(@TempDir java.nio.file.Path dir) throws Exception {
        List<double[]> rows = threeByFour();
        writeParquet(dir, rows, Schema.Type.FLOAT);

        List<WeightedPoint> read = readAll(source(dir, Map.of()));

        assertEquals(rows.size(), read.size());
        assertEquals(4, read.get(0).size());
        for (WeightedPoint v : read) {
            assertEquals(4, v.values().length);
        }
    }

    @Test
    void sampleFractionDropsRowsAndIsReproducible(@TempDir java.nio.file.Path dir) throws Exception {
        List<double[]> rows = new ArrayList<>();
        for (int i = 0; i < 2000; i++) {
            rows.add(new double[] {i, 0.0});
        }
        writeParquet(dir, rows, Schema.Type.DOUBLE);

        DataSource sampled = source(dir, Map.of("sampleFraction", 0.25, "seed", 7L));
        int first = readAll(sampled).size();
        int second = readAll(source(dir, Map.of("sampleFraction", 0.25, "seed", 7L))).size();

        assertEquals(first, second, "a fixed seed and parallelism must draw the same subset size");
        assertTrue(first > 300 && first < 700,
            "expected roughly a quarter of 2000 rows, got " + first);
        assertEquals(2000, readAll(source(dir, Map.of())).size(), "no fraction = every row");
    }

    @Test
    void metadataRecordsWhatWasRead(@TempDir java.nio.file.Path dir) throws Exception {
        writeParquet(dir, threeByFour(), Schema.Type.DOUBLE);
        Map<String, String> metadata =
            source(dir, Map.of("sampleFraction", 0.5, "numPartitions", 8)).metadata();

        assertEquals("parquet", metadata.get("format"));
        assertEquals("features", metadata.get("featureColumnName"));
        assertEquals("0.5", metadata.get("sampleFraction"));
        assertEquals("8", metadata.get("numPartitions"));
        assertEquals("none", metadata.get("weightColumn"));
    }

    @Test
    void aMissingFeatureColumnIsRejectedWithTheAvailableColumns(@TempDir java.nio.file.Path dir)
            throws Exception {
        writeParquet(dir, threeByFour(), Schema.Type.DOUBLE);
        Map<String, Object> params = new HashMap<>();
        params.put("path", dir.toString());
        params.put("featureColumnName", "emb");
        DataSource dataSource = ParquetDataSource.factory().create(params);

        String message = assertThrows(IllegalArgumentException.class,
            () -> readAll(dataSource)).getMessage();
        assertTrue(message.contains("emb") && message.contains("features"),
            "the error must name both the missing and the available columns, got: " + message);
    }

    /** {@code weightColumn} is the per-row multiplicity. Absent means 1.0, so an old config keeps
     *  its meaning exactly; present, it has to arrive on the record. */
    @Test
    void readsAnOptionalWeightColumn(@TempDir java.nio.file.Path dir) throws Exception {
        Schema schema = SchemaBuilder.record("Row").fields()
            .name("features").type(Schema.createArray(Schema.create(Schema.Type.DOUBLE))).noDefault()
            .name("w").type().doubleType().noDefault()
            .endRecord();
        Path file = new Path(dir.resolve("part-00000.parquet").toString());
        try (ParquetWriter<GenericRecord> writer =
                 AvroParquetWriter.<GenericRecord>builder(file).withSchema(schema).build()) {
            for (int i = 0; i < 20; i++) {
                GenericRecord record = new GenericData.Record(schema);
                record.put("features", List.of((double) i, 0.0));
                record.put("w", i + 1.0);
                writer.write(record);
            }
        }

        Map<String, Object> params = new HashMap<>();
        params.put("path", dir.toString());
        params.put("featureColumnName", "features");
        params.put("weightColumn", "w");
        List<WeightedPoint> read = readAll(ParquetDataSource.factory().create(params));

        assertEquals(20, read.size());
        double totalMass = 0.0;
        for (WeightedPoint p : read) {
            // Weight follows the row: this file sets w = x + 1.
            assertEquals(p.values()[0] + 1.0, p.weight, 1e-12);
            totalMass += p.weight;
        }
        assertEquals(210.0, totalMass, 1e-9, "sum of 1..20");

        // No weight column named -> unit weights, so the row count IS the mass.
        for (WeightedPoint p : readAll(source(dir, Map.of()))) {
            assertEquals(1.0, p.weight, 1e-12);
        }
    }

    @Test
    void aMissingWeightColumnIsRejected(@TempDir java.nio.file.Path dir) throws Exception {
        writeParquet(dir, threeByFour(), Schema.Type.DOUBLE);
        Map<String, Object> params = new HashMap<>();
        params.put("path", dir.toString());
        params.put("featureColumnName", "features");
        params.put("weightColumn", "nope");
        DataSource dataSource = ParquetDataSource.factory().create(params);

        assertTrue(assertThrows(IllegalArgumentException.class, () -> readAll(dataSource))
            .getMessage().contains("nope"));
    }

    @Test
    void pathAndFeatureColumnAreRequired() {
        assertTrue(assertThrows(IllegalArgumentException.class,
            () -> ParquetDataSource.factory().create(new HashMap<>()))
            .getMessage().contains("path"));

        Map<String, Object> onlyPath = new HashMap<>();
        onlyPath.put("path", "/tmp/nowhere");
        assertTrue(assertThrows(IllegalArgumentException.class,
            () -> ParquetDataSource.factory().create(onlyPath))
            .getMessage().contains("featureColumnName"));
    }
}
