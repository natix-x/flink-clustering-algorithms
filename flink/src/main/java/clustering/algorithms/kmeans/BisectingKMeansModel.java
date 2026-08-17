package clustering.algorithms.kmeans;

import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Model;
import clustering.distance.DistanceMetric;

/** A fitted bisecting k-means model: a binary cluster tree, stored as flat arrays
 *  ({@code node 0} is the root, {@code left[i] &lt; 0} marks a leaf). Java counterpart of the
 *  Spark {@code BisectingKMeansModel}, whose tree is a sealed-trait node graph — flat arrays
 *  here because the tree also travels the FLIP-176 feedback edge during fitting, and arrays
 *  of primitives serialise without a recursive-object round-trip.
 *
 *  Labelling is <b>root-to-leaf traversal</b>, not nearest-leaf-centroid: at each internal
 *  node the point follows the closer child centroid. This matches the Spark side (and MLlib's
 *  {@code BisectingKMeansModel}) and costs O(depth) distance computations per point instead of
 *  O(k), but it can differ from a flat nearest-centroid assignment — a point may end up in a
 *  leaf whose centroid is not globally closest. That is a property of divisive hierarchical
 *  clustering, not a bug.
 *
 *  Ties go left, so labelling is deterministic. */
public class BisectingKMeansModel implements Model {

    private final double[][] centroids;
    private final int[] left;
    private final int[] right;
    /** Cluster id per node; {@code -1} for internal nodes. Numbered in left-to-right DFS
     *  order, so ids depend only on the tree shape. */
    private final int[] leafId;
    private final DistanceMetric distance;

    public BisectingKMeansModel(double[][] centroids, int[] left, int[] right, int[] leafId,
                                DistanceMetric distance) {
        this.centroids = centroids;
        this.left = left;
        this.right = right;
        this.leafId = leafId;
        this.distance = distance;
    }

    @Override
    public int predict(DenseVector features) {
        double[] point = features.values;
        int node = 0;
        while (left[node] >= 0) {
            int l = left[node];
            int r = right[node];
            node = distance.compute(point, centroids[l]) <= distance.compute(point, centroids[r]) ? l : r;
        }
        return leafId[node];
    }

    public int numClusters() {
        int n = 0;
        for (int id : leafId) {
            if (id >= 0) {
                n++;
            }
        }
        return n;
    }

    /** Leaf centroids in cluster-id order. */
    public double[][] clusterCentroids() {
        double[][] out = new double[numClusters()][];
        for (int node = 0; node < leafId.length; node++) {
            if (leafId[node] >= 0) {
                out[leafId[node]] = centroids[node];
            }
        }
        return out;
    }
}
