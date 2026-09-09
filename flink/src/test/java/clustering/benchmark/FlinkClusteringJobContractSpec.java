package clustering.benchmark;

import clustering.benchmark.config.ClusterProfile;
import clustering.benchmark.config.RunConfig;
import clustering.benchmark.metrics.RunResult;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.networknt.schema.JsonSchemaFactory;
import com.networknt.schema.SpecVersion;
import com.networknt.schema.ValidationMessage;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** End-to-end contract test: actually run {@link FlinkClusteringJob} on a tiny synthetic dataset
 *  and assert the REAL emitted RunResult validates against {@code run_result.schema.json}
 *  (vendored as a test resource, byte-identical to the Spark engine's copy).
 *
 *  This exercises the real fill logic (metric-reporter snapshot, timings, eval metrics,
 *  serialisation) — the thing that runs when you launch the benchmark. It also guards Java/schema
 *  drift: the schema is {@code additionalProperties: false}, so a new RunResult field not declared
 *  in the schema fails here rather than in a SLURM array job.
 *
 *  <p>Mirrors the Spark {@code SparkClusteringJobContractSpec}. The schema only declares
 *  counters BOTH engines emit on the same measurement basis — Spark-only counters were cut
 *  from the contract rather than kept as an always-null placeholder. That is the point of
 *  having ONE contract: the harness reads either engine's file with the same parser, and
 *  every field present actually means the same thing on both sides.
 *
 *  <p>The failure path is checked too: a failed run must STILL emit a schema-valid result
 *  ({@code status: "failed"} + {@code errorMessage}), because the SLURM array jobs rely on it to
 *  never lose a run silently. */
class FlinkClusteringJobContractSpec {

    private static String config(String runId, String algorithm, String algorithmParams) {
        return "{"
            + "\"runId\":\"" + runId + "\","
            + "\"profile\":\"local\","
            + "\"dataset\":{\"type\":\"synthetic\","
            + "\"params\":{\"numPoints\":800,\"numPartitions\":2,\"seed\":42}},"
            + "\"algorithm\":{\"name\":\"" + algorithm + "\","
            + "\"params\":{" + algorithmParams + "}},"
            + "\"evaluation\":{\"metrics\":[\"silhouette\",\"nClusters\",\"clusterSizes\","
            + "\"noiseFraction\"],\"sampleSize\":300,\"seed\":42}"
            + "}";
    }

    private static void assertConformsToSchema(RunResult result) throws Exception {
        InputStream schemaStream =
            FlinkClusteringJobContractSpec.class.getResourceAsStream("/run_result.schema.json");
        assertNotNull(schemaStream, "run_result.schema.json missing from test resources");

        String json = RunResult.toJsonString(result);
        Set<ValidationMessage> violations = JsonSchemaFactory
            .getInstance(SpecVersion.VersionFlag.V202012)
            .getSchema(schemaStream)
            .validate(new ObjectMapper().readTree(json));

        assertTrue(violations.isEmpty(),
            "RunResult violates the contract schema:\n" + violations + "\n\njson:\n" + json);
    }

    private static RunResult run(String json) throws Exception {
        return new FlinkClusteringJob().run(
            RunConfig.fromJsonString(json), new ClusterProfile.LocalProfile());
    }

    /** One entry per algorithm FAMILY, since the families fill different parts of the result:
     *  k-means has no {@code distance} param, the medoid ladder has no noise label, and DBSCAN++
     *  is the only entry that can emit one. */
    @Test
    void everyAlgorithmFamilyEmitsASchemaValidResult() throws Exception {
        String[][] entries = {
            {"contract-kmeans", "kmeans", "\"k\":4,\"maxIter\":5,\"seed\":42"},
            {"contract-bisecting", "bisectingkmeans",
                "\"k\":4,\"maxIter\":5,\"seed\":42,\"trials\":2,\"select\":\"size\""},
            {"contract-fastpam", "fastpam", "\"k\":4,\"maxIter\":5,\"distance\":\"euclidean\""},
            {"contract-fasterpam", "fasterpam",
                "\"k\":4,\"maxIter\":5,\"distance\":\"euclidean\",\"seed\":42"},
            {"contract-clara", "clara", "\"k\":4,\"numSamples\":2,\"sampleSize\":200,"
                + "\"maxIter\":5,\"distance\":\"euclidean\",\"inner\":\"fasterpam\",\"seed\":42"},
            {"contract-pamae", "pamae", "\"k\":4,\"numSamples\":2,\"sampleSize\":200,"
                + "\"maxIter\":5,\"refineIters\":2,\"poolSize\":150,"
                + "\"distance\":\"euclidean\",\"seed\":42"},
            {"contract-dbscanpp", "dbscanpp", "\"eps\":3.0,\"minPts\":5,"
                + "\"coreSampleFraction\":0.5,\"distance\":\"euclidean\",\"seed\":42"}
        };

        for (String[] entry : entries) {
            RunResult r = run(config(entry[0], entry[1], entry[2]));
            assertEquals("ok", r.status, entry[1] + " should succeed; error=" + r.errorMessage);
            assertEquals("flink", r.framework, entry[1]);
            assertEquals(entry[0], r.runId);
            assertConformsToSchema(r);
        }
    }

    /** A failed run must still produce a schema-valid result, or a SLURM array job loses it. */
    @Test
    void aFailedRunStillEmitsASchemaValidResult() throws Exception {
        // k greater than the dataset: the medoid ladder rejects it, so the job fails for a real
        // reason rather than a fabricated one.
        RunResult r = run(config("contract-failed", "fastpam",
            "\"k\":9999,\"maxIter\":5,\"distance\":\"euclidean\""));

        assertEquals("failed", r.status);
        assertNotNull(r.errorMessage, "a failed run must say why");
        assertConformsToSchema(r);
    }

    /** {@code distance} is REQUIRED on every entry that reads it — the same rule Spark enforces.
     *  A config that omits it must fail loudly rather than silently run euclidean, otherwise the
     *  two engines could disagree on what a shared experiment config means. */
    @Test
    void aMissingDistanceParamFailsTheRun() throws Exception {
        RunResult r = run(config("contract-nodistance", "fastpam", "\"k\":4,\"maxIter\":5"));

        assertEquals("failed", r.status, "a medoid run without 'distance' must not default silently");
        assertTrue(r.errorMessage.contains("distance"),
            "the error must name the missing param, got: " + r.errorMessage);
        assertConformsToSchema(r);
    }
}