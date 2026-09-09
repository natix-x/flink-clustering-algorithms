package clustering.core;

import java.util.Arrays;

/** Point weights: {@code w_x} = how many points {@code x} stands for, so every Σ-over-points in
 *  the repo optimises the objective of the data the input represents rather than of the rows it
 *  happens to contain. Java mirror of the Spark {@code clustering.core.Weights}.
 *
 *  Absent weights mean 1.0 — an unweighted run is the weighted run of a unit-weight input, which
 *  is why nothing in the code has to branch on "is this weighted". */
public final class Weights {

    private Weights() {}

    /** Unit weights for {@code count} points. */
    public static double[] unit(int count) {
        double[] weights = new double[count];
        Arrays.fill(weights, 1.0);
        return weights;
    }

    /** A weight that can safely enter a Σ: NaN, negative and infinite become 0.0, i.e. the row
     *  carries no points. Java counterpart of the Spark {@code Weights.safeColumn}, and the two
     *  must agree — the evaluators divide by a mass they also sum, so a value that survives into
     *  one and not the other produces a score no reader can check.
     *
     *  Neutralising rather than failing is deliberate: a single corrupt row in a 10^9-row parquet
     *  file should cost its own row, not the run. What it costs is reported —
     *  {@code silhouetteUnscoredPoints} counts the rows the evaluator refused. A negative weight is
     *  the dangerous one: it makes its cluster's mean distance negative, which wins the {@code min}
     *  that produces {@code b} for EVERY point and drags the whole score toward -1. */
    public static double safe(double weight) {
        return (Double.isFinite(weight) && weight > 0.0) ? weight : 0.0;
    }
}
