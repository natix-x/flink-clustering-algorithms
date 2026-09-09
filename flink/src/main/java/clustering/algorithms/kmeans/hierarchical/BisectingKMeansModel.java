package clustering.algorithms.kmeans.hierarchical;

import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Model;
import clustering.distance.DistanceMetric;

/**
 * Fitted bisecting k-means model represented as a binary cluster tree using flat arrays.
 * Prediction uses root-to-leaf traversal, following the closest child centroid at each internal node.
 */
public class BisectingKMeansModel implements Model {

    private final DenseVector[] centroids;
    private final int[] leftChildren;
    private final int[] rightChildren;

    /** Cluster ID per node; -1 for internal nodes. */
    private final int[] leafIds;
    private final DistanceMetric distanceMetric;

    public BisectingKMeansModel(DenseVector[] centroids, int[] leftChildren, int[] rightChildren, int[] leafIds,
                                DistanceMetric distanceMetric) {
        this.centroids = centroids;
        this.leftChildren = leftChildren;
        this.rightChildren = rightChildren;
        this.leafIds = leafIds;
        this.distanceMetric = distanceMetric;
    }

    @Override
    public int predict(DenseVector features) {
        int currentNode = 0;
        while (leftChildren[currentNode] >= 0) {
            int leftChild = leftChildren[currentNode];
            int rightChild = rightChildren[currentNode];

            double leftDist = distanceMetric.compute(features, centroids[leftChild]);
            double rightDist = distanceMetric.compute(features, centroids[rightChild]);

            currentNode = leftDist <= rightDist ? leftChild : rightChild;
        }
        return leafIds[currentNode];
    }

    public int getClusterCount() {
        int count = 0;
        for (int id : leafIds) {
            if (id >= 0) {
                count++;
            }
        }
        return count;
    }

    /** Returns leaf centroids ordered by their cluster ID. */
    public DenseVector[] getLeafCentroids() {
        DenseVector[] leafCentroids = new DenseVector[getClusterCount()];
        for (int nodeId = 0; nodeId < leafIds.length; nodeId++) {
            if (leafIds[nodeId] >= 0) {
                leafCentroids[leafIds[nodeId]] = centroids[nodeId];
            }
        }
        return leafCentroids;
    }
}
