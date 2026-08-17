package clustering.algorithms.dbscan;

import org.apache.flink.ml.linalg.DenseVector;
import clustering.core.Model;
import clustering.distance.DistanceMetric;

import java.util.Arrays;

/** A fitted density model: labelled core points, plus the rule that labels everything else.
 *  Java mirror of the Spark {@code clustering.algorithms.dbscan.CoreLabelModel}.
 *
 *  Assignment is one nearest-core scan per point — the core set is at most m points, so
 *  labelling needs no join and no second distributed pass; on Flink the model simply travels
 *  with the operator that labels the stream.
 *
 *  @param corePoints        core-point coordinates
 *  @param coreClusterLabels cluster id of each core point; contiguous from 0, numbered by
 *                           ascending smallest candidate index, so independent of partitioning
 *  @param requireWithinEps  {@code true} (config {@code assign: eps}) = a point joins its
 *                           closest core only if that core is within eps, else {@code -1} —
 *                           classic DBSCAN noise semantics. {@code false} ({@code assign:
 *                           closest}) = the paper's rule, which assigns every point and so
 *                           emits no noise at all. */
public class CoreLabelModel implements Model {

    private final double[][] corePoints;
    private final int[] coreClusterLabels;
    private final double eps;
    private final DistanceMetric distanceMetric;
    private final boolean requireWithinEps;

    public CoreLabelModel(double[][] corePoints, int[] coreClusterLabels, double eps,
                          DistanceMetric distanceMetric, boolean requireWithinEps) {
        if (corePoints.length != coreClusterLabels.length) {
            throw new IllegalArgumentException("cores (" + corePoints.length + ") and labels ("
                + coreClusterLabels.length + ") must have the same length");
        }
        this.corePoints = corePoints;
        this.coreClusterLabels = coreClusterLabels;
        this.eps = eps;
        this.distanceMetric = distanceMetric;
        this.requireWithinEps = requireWithinEps;
    }

    /** Number of clusters found; 0 when the parameters produced no core point at all. */
    public int numClusters() {
        return (int) Arrays.stream(coreClusterLabels).distinct().count();
    }

    public double[][] corePoints() {
        return corePoints;
    }

    public int[] coreClusterLabels() {
        return coreClusterLabels;
    }

    @Override
    public int predict(DenseVector features) {
        double[] point = features.values;
        int nearestCore = -1;
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < corePoints.length; i++) {
            double d = distanceMetric.compute(point, corePoints[i]);
            if (d < minDistance) {
                minDistance = d;
                nearestCore = i;
            }
        }
        boolean outsideEpsilon = requireWithinEps && minDistance > eps;
        return (nearestCore < 0 || outsideEpsilon) ? -1 : coreClusterLabels[nearestCore];
    }
}
