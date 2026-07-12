package clustering.evaluation;

import clustering.distance.DistanceMetric;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Mean silhouette over non-noise points: s(p) = (b - a) / max(a, b).
 *  O(n^2); the caller samples the data down first. Mirrors the Spark
 *  {@code SilhouetteEvaluator} semantics (noise label -1 excluded). */
public final class SilhouetteEvaluator {

    private final DistanceMetric distance;

    public SilhouetteEvaluator(DistanceMetric distance) {
        this.distance = distance;
    }

    /** @param points index-aligned with {@code labels}. */
    public double evaluate(List<double[]> points, int[] labels) {
        Map<Integer, List<double[]>> clusters = new HashMap<>();
        for (int i = 0; i < points.size(); i++) {
            int label = labels[i];
            if (label == -1) continue;
            clusters.computeIfAbsent(label, k -> new ArrayList<>()).add(points.get(i));
        }
        if (clusters.isEmpty()) return 0.0;

        double scoreSum = 0.0;
        long cnt = 0;
        for (int i = 0; i < points.size(); i++) {
            int label = labels[i];
            if (label == -1) continue;
            double[] p = points.get(i);
            double a = meanDistExclSelf(p, clusters.get(label));
            double b = Double.MAX_VALUE;
            for (Map.Entry<Integer, List<double[]>> e : clusters.entrySet()) {
                if (e.getKey() == label) continue;
                b = Math.min(b, meanDist(p, e.getValue()));
            }
            double denom = Math.max(a, b);
            scoreSum += (denom == 0.0) ? 0.0 : (b - a) / denom;
            cnt++;
        }
        return cnt == 0 ? 0.0 : scoreSum / cnt;
    }

    private double meanDist(double[] p, List<double[]> pts) {
        if (pts.isEmpty()) return 0.0;
        double s = 0.0;
        for (double[] q : pts) s += distance.compute(p, q);
        return s / pts.size();
    }

    private double meanDistExclSelf(double[] p, List<double[]> pts) {
        double s = 0.0;
        int cnt = 0;
        for (double[] q : pts) {
            if (q != p) {
                s += distance.compute(p, q);
                cnt++;
            }
        }
        return cnt == 0 ? 0.0 : s / cnt;
    }
}
