package clustering.benchmark.evaluation;

import java.io.Serializable;

/** Per-label count, shuffled in the distributed cluster-sizes job. POJO. */
public class LabelCount implements Serializable {
    public int label;
    public long count;

    public LabelCount() {}
}