package clustering.benchmark;

import clustering.benchmark.config.ClusterProfile;
import clustering.benchmark.config.RunConfig;
import clustering.benchmark.framework.FlinkClusteringJob;
import clustering.benchmark.metrics.RunResult;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** End-to-end smoke test: runs KMeans (Flink ML iterations) on a local Flink
 *  MiniCluster and checks the produced RunResult. */
class FlinkClusteringJobTest {

    private static final String CONFIG_JSON =
        "{"
        + "\"runId\":\"test-kmeans-synth\","
        + "\"profile\":\"local\","
        + "\"dataset\":{\"type\":\"synthetic\",\"params\":{\"numPoints\":2000,\"numPartitions\":2,\"seed\":42}},"
        + "\"algorithm\":{\"name\":\"kmeans\",\"params\":{\"k\":5,\"maxIter\":20,\"distance\":\"euclidean\"}},"
        + "\"evaluation\":{\"metrics\":[\"silhouette\",\"nClusters\",\"clusterSizes\",\"noiseFraction\"],\"sampleSize\":500,\"seed\":42}"
        + "}";

    @Test
    void runsKMeansEndToEnd() throws Exception {
        RunConfig config = RunConfig.fromJsonString(CONFIG_JSON);
        RunResult r = new FlinkClusteringJob().run(config, new ClusterProfile.LocalProfile());

        assertEquals("ok", r.status, "run should succeed; error=" + r.errorMessage);
        assertEquals("flink", r.framework);
        assertEquals(2000L, r.nRows, "sizes should cover all points");
        assertEquals(Integer.valueOf(3), r.nFeatures);
        assertTrue(r.nClusters > 0, "should find clusters");
        assertNotNull(r.clusterSizes);
        assertTrue(r.clusterSizes.size() > 0);
        assertNotNull(r.silhouette, "silhouette requested -> should be present");

        String json = RunResult.toJsonString(r);
        assertTrue(json.contains("\"framework\":\"flink\""));
        assertTrue(json.contains("\"engineConf\""));

        RunResult.writeToDir(r, "target/contract-check");
    }

    private static final String DISTFASTPAM_JSON =
        "{"
        + "\"runId\":\"test-distfastpam-synth\","
        + "\"profile\":\"local\","
        + "\"dataset\":{\"type\":\"synthetic\",\"params\":{\"numPoints\":600,\"numPartitions\":2,\"seed\":42}},"
        + "\"algorithm\":{\"name\":\"distfastpam\",\"params\":{\"k\":4,\"maxIter\":20,\"distance\":\"euclidean\"}},"
        + "\"evaluation\":{\"metrics\":[\"clusterSizes\",\"noiseFraction\"],\"seed\":42}"
        + "}";

    @Test
    void runsDistributedFastPamEndToEnd() throws Exception {
        RunConfig config = RunConfig.fromJsonString(DISTFASTPAM_JSON);
        RunResult r = new FlinkClusteringJob().run(config, new ClusterProfile.LocalProfile());

        assertEquals("ok", r.status, "run should succeed; error=" + r.errorMessage);
        assertEquals(600L, r.nRows);
        assertNotNull(r.clusterSizes);

        // nClusters / silhouette NOT requested in this config -> omitted (null), matching
        // Spark's per-metric gating; the cluster count is derived from clusterSizes instead.
        assertNull(r.nClusters, "nClusters not requested -> should be omitted");
        assertNull(r.silhouette, "silhouette not requested -> should be omitted");

        long nonNoiseClusters = r.clusterSizes.keySet().stream()
            .filter(k -> Integer.parseInt(k) >= 0).count();
        assertEquals(4L, nonNoiseClusters, "FastPAM should produce exactly k=4 non-empty-eligible clusters");

        long covered = r.clusterSizes.values().stream().mapToLong(Long::longValue).sum();
        assertEquals(600L, covered, "every point assigned to a medoid");
    }
}