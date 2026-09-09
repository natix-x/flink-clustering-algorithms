package clustering.algorithms.dbscan.components;

import clustering.core.EnvFactory;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;
import java.io.Serializable;

/** How the m candidate core points are chosen — the knob that decides how much of the accuracy
 *  DBSCAN++ keeps for a given cost.
 *
 *  Uniform random subsampling (Jang &amp; Jiang, ICML 2019) is the sole strategy: it is P2 — one
 *  Flink job whose result is collected to the driver, from where the counting job ships it to
 *  every subtask.
 *
 *  Selection is over ROWS: candidates only have to cover the space, and the densities that
 *  decide core-point status are counted afterwards against the full dataset. */
public interface CandidateSelectionStrategy extends Serializable {

    String strategyName();

    /** Selects at most {@code targetCandidateCount} candidates from the source. */
    double[][] selectCandidates(
        PointSource source,
        EnvFactory envs,
        long totalRowCount,
        int targetCandidateCount,
        long seed,
        DistanceMetric distanceMetric);

    /** Resolves a {@code sampling} param value. */
    static CandidateSelectionStrategy fromName(String name) {
        switch (name.toLowerCase()) {
            case "uniform":
                return UniformSelection.INSTANCE;
            default:
                throw new IllegalArgumentException(
                    "Unknown candidate sampling: '" + name + "'. Known: uniform");
        }
    }
}
