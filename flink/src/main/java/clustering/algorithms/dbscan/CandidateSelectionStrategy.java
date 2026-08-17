package clustering.algorithms.dbscan;

import clustering.core.EnvFactory;
import clustering.core.PointSource;
import clustering.distance.DistanceMetric;

import java.io.Serializable;

/** How the m candidate core points are chosen — the knob that decides how much of the accuracy
 *  DBSCAN++ keeps for a given cost. Java mirror of the Spark
 *  {@code clustering.algorithms.dbscan.CandidateSelectionStrategy}.
 *
 *  The paper (Jang &amp; Jiang, ICML 2019) offers uniform random and greedy K-center (a
 *  2-approximation, generally better in their experiments); {@code linspace} comes from the
 *  Spark DBSCAN++ realisation (IJDSA 2026) and is the cheapest of the three. All three are P2:
 *  one Flink job whose result is collected to the driver, from where the counting job ships it
 *  to every subtask.
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

    /** Resolves a {@code sampling} param value. {@code kcenter} also reads {@code poolFactor}. */
    static CandidateSelectionStrategy fromName(String name, int poolFactor) {
        switch (name.toLowerCase()) {
            case "uniform":
                return UniformSelection.INSTANCE;
            case "linspace":
                return LinspaceSelection.INSTANCE;
            case "kcenter":
                return new KCenterSelection(poolFactor);
            default:
                throw new IllegalArgumentException(
                    "Unknown candidate sampling: '" + name + "'. Known: kcenter, linspace, uniform");
        }
    }
}
