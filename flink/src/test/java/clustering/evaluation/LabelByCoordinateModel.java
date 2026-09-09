package clustering.evaluation;

import clustering.core.Model;
import org.apache.flink.ml.linalg.DenseVector;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/** A {@link Model} whose labelling is fixed by the test, keyed by the point's coordinates.
 *
 *  <p>The Spark specs do this with a label COLUMN the frame already carries; the Flink seam is
 *  {@code predict(DenseVector)}, which has no row identity, so the labelling has to be a function
 *  of the coordinates. That is the whole difference: what is under test either way is the index
 *  arithmetic and the distributed moment/draw jobs, never a clusterer.
 *
 *  <p>Consequence worth knowing when porting a Spark case: two rows with the SAME coordinates in
 *  DIFFERENT clusters cannot be expressed here. The one Spark spec that needs it (Davies-Bouldin
 *  with coincident centroids) is asserted against hand-built {@link ClusterMoments} instead. */
final class LabelByCoordinateModel implements Model {

    private final Map<String, Integer> labelByPoint = new HashMap<>();

    /** @param rows alternating point / label: {@code {0.0}, 0, {10.0}, 1, ...} is not expressible
     *              in Java varargs, so points and labels are passed as two aligned arrays. */
    LabelByCoordinateModel(double[][] points, int[] labels) {
        if (points.length != labels.length) {
            throw new IllegalArgumentException("points and labels must be index-aligned");
        }
        for (int i = 0; i < points.length; i++) {
            labelByPoint.put(key(points[i]), labels[i]);
        }
    }

    @Override
    public int predict(DenseVector features) {
        Integer label = labelByPoint.get(key(features.values));
        if (label == null) {
            throw new IllegalStateException("no fixture label for " + Arrays.toString(features.values));
        }
        return label;
    }

    private static String key(double[] coordinates) {
        return Arrays.toString(coordinates);
    }
}
