package clustering.benchmark;

import clustering.algorithms.kmedoids.local.FastPAM;
import clustering.benchmark.config.ClusterProfile;
import clustering.benchmark.config.RunConfig;
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

    private static final String BISECTING_JSON =
        "{"
        + "\"runId\":\"test-bisectingkmeans-synth\","
        + "\"profile\":\"local\","
        + "\"dataset\":{\"type\":\"synthetic\",\"params\":{\"numPoints\":2000,\"numPartitions\":2,\"seed\":42}},"
        + "\"algorithm\":{\"name\":\"bisectingkmeans\",\"params\":{\"k\":4,\"maxIter\":10}},"
        + "\"evaluation\":{\"metrics\":[\"nClusters\",\"clusterSizes\"],\"seed\":42}"
        + "}";

    @Test
    void runsBisectingKMeansEndToEnd() throws Exception {
        RunConfig config = RunConfig.fromJsonString(BISECTING_JSON);
        RunResult r = new FlinkClusteringJob().run(config, new ClusterProfile.LocalProfile());

        assertEquals("ok", r.status, "run should succeed; error=" + r.errorMessage);
        assertEquals(Integer.valueOf(4), r.nClusters, "k leaves requested and available");
        long covered = r.clusterSizes.values().stream().mapToLong(Long::longValue).sum();
        assertEquals(2000L, covered, "every point lands in a leaf");
    }

    /** DBSCAN++ is the only entry that can emit the noise label, so the e2e checks that the
     *  noise plumbing (label -1 -> noiseFraction) works end to end on the synthetic mixture,
     *  whose 5% uniform component is genuine noise at this eps. */
    private static final String DBSCANPP_JSON =
        "{"
        + "\"runId\":\"test-dbscanpp-synth\","
        + "\"profile\":\"local\","
        + "\"dataset\":{\"type\":\"synthetic\",\"params\":{\"numPoints\":800,\"numPartitions\":2,\"seed\":42}},"
        + "\"algorithm\":{\"name\":\"dbscanpp\",\"params\":{\"eps\":3.0,\"minPts\":5,"
        + "\"coreSampleFraction\":0.5,\"sampling\":\"uniform\",\"distance\":\"euclidean\",\"seed\":42}},"
        + "\"evaluation\":{\"metrics\":[\"nClusters\",\"clusterSizes\",\"noiseFraction\"],\"seed\":42}"
        + "}";

    @Test
    void runsDbscanPlusPlusEndToEnd() throws Exception {
        RunConfig config = RunConfig.fromJsonString(DBSCANPP_JSON);
        RunResult r = new FlinkClusteringJob().run(config, new ClusterProfile.LocalProfile());

        assertEquals("ok", r.status, "run should succeed; error=" + r.errorMessage);
        assertTrue(r.nClusters > 0, "the dense modes must form clusters");
        assertNotNull(r.noiseFraction);
        assertTrue(r.noiseFraction > 0.0 && r.noiseFraction < 1.0,
            "the uniform component must be noise, the dense modes must not: " + r.noiseFraction);
        long covered = r.clusterSizes.values().stream().mapToLong(Long::longValue).sum();
        assertEquals(800L, covered, "every point is labelled or marked noise");
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

        // `silhouette` was not requested, so it is omitted — per-metric gating, as on Spark.
        assertNull(r.silhouette, "silhouette not requested -> should be omitted");

        // `nClusters` is the ONE exception to that gating, on both engines: it is emitted whenever
        // the label-stats scan ran at all, which requesting clusterSizes/noiseFraction/silhouette
        // does. The count is already in the scan, and gating it would cost
        // `silhouetteSampleClusters` its meaning — "87 clusters in the sample" says nothing until
        // you know whether the labelling had 87 or 100.
        assertEquals(4, r.nClusters, "the label-stats scan ran, so nClusters comes for free");

        long nonNoiseClusters = r.clusterSizes.keySet().stream()
            .filter(k -> Integer.parseInt(k) >= 0).count();
        assertEquals(4L, nonNoiseClusters, "FastPAM should produce exactly k=4 non-empty-eligible clusters");

        long covered = r.clusterSizes.values().stream().mapToLong(Long::longValue).sum();
        assertEquals(600L, covered, "every point assigned to a medoid");
    }
}
