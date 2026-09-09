package clustering.core;

import org.apache.flink.ml.linalg.DenseVector;

/** The stream RECORD: a point plus how many points it stands for. Java counterpart of the Spark
 *  side's {@code [features, weight]} DataFrame row.
 *
 *  <h3>Why the weight is in the record and not a column</h3>
 *  Spark can express "weight defaults to 1.0" as a constant Catalyst expression ({@code lit(1.0)}),
 *  which costs nothing per row — the projection is folded away. Flink's DataStream has no notion of
 *  a constant column, so an absent weight has to be materialised as a real 1.0 in every record.
 *  Measured in bytes: {@link WeightedPointSerializer} writes 8 more per record than a bare
 *  {@link DenseVector}, i.e. +13% at 8 dimensions (Gaia, NYC — the two largest sets by row count)
 *  and +0.1% at 1024 (Cohere). That asymmetry is a finding about the platforms, not an
 *  implementation detail: the same feature is free on one engine and costs 13% of the wire and
 *  cache volume on the other.
 *
 *  <h3>Features stay a DenseVector</h3>
 *  {@link #features} is the ecosystem's point type, the counterpart of Spark's {@code VectorUDT},
 *  and it is the type the fitted models and prototypes use. It is still a RECORD-level wrapper
 *  only: every operator reads {@code features.values} — a field read, not a copy — before its hot
 *  loop, so both engines keep running the identical raw-array distance kernel. Flattening it to a
 *  bare {@code double[]} would save one object header per deserialised record and NOT a single
 *  serialised byte; the serializer's reuse path already removes most of that, so the wrapper stays.
 *
 *  <h3>Weight semantics</h3>
 *  A point of weight w contributes w times to every Σ-over-points, so a weighted input optimises
 *  the objective of the data it represents. The repo-wide invariant is that <b>weighting equals
 *  duplication</b>: clustering one row of weight 3 must equal clustering three copies of it. Weights
 *  may be fractional (lightweight coresets use w = 1/(m·q(x))), so a weight is never a row count.
 *
 *  Public mutable fields and a no-arg constructor because Flink instantiates and reuses records. */
public final class WeightedPoint {

    public DenseVector features;
    public double weight;

    public WeightedPoint() {}

    public WeightedPoint(DenseVector features, double weight) {
        this.features = features;
        this.weight = weight;
    }

    /** Unit-weight record — an unweighted point is the weighted point of weight 1. */
    public static WeightedPoint of(DenseVector features) {
        return new WeightedPoint(features, 1.0);
    }

    public static WeightedPoint of(double[] features) {
        return new WeightedPoint(new DenseVector(features), 1.0);
    }

    /** The coordinate array — the vector's OWN array, so this copies nothing. */
    public double[] values() {
        return features.values;
    }

    public int size() {
        return features.size();
    }

    @Override
    public String toString() {
        return "WeightedPoint{" + features + ", w=" + weight + '}';
    }
}
