package clustering.core;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.TypeSerializer;

/** {@link TypeInformation} for {@link WeightedPoint}, so the stream record gets
 *  {@link WeightedPointSerializer} instead of the reflective POJO serializer or the Kryo fallback.
 *
 *  Mirrors how Flink ML publishes {@code DenseVectorTypeInfo} for its own point type: the record
 *  type of a pipeline should never be left to type extraction, because the fallback silently costs
 *  per-record reflection and, for Kryo, breaks state compatibility. */
public final class WeightedPointTypeInfo extends TypeInformation<WeightedPoint> {

    public static final WeightedPointTypeInfo INSTANCE = new WeightedPointTypeInfo();

    private static final long serialVersionUID = 1L;

    private WeightedPointTypeInfo() {}

    @Override
    public boolean isBasicType() {
        return false;
    }

    @Override
    public boolean isTupleType() {
        return false;
    }

    /** One logical field: the record is treated as an opaque point, never split by position. */
    @Override
    public int getArity() {
        return 1;
    }

    @Override
    public int getTotalFields() {
        return 1;
    }

    @Override
    public Class<WeightedPoint> getTypeClass() {
        return WeightedPoint.class;
    }

    /** Not a key type: nothing in the repo keys BY a point (the keyed reduces key by a constant to
     *  funnel partials into one task), and declaring it comparable would invite a shuffle that no
     *  algorithm here wants. */
    @Override
    public boolean isKeyType() {
        return false;
    }

    @Override
    public TypeSerializer<WeightedPoint> createSerializer(ExecutionConfig config) {
        return WeightedPointSerializer.INSTANCE;
    }

    @Override
    public String toString() {
        return "WeightedPoint";
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof WeightedPointTypeInfo;
    }

    @Override
    public int hashCode() {
        return WeightedPointTypeInfo.class.hashCode();
    }

    @Override
    public boolean canEqual(Object other) {
        return other instanceof WeightedPointTypeInfo;
    }
}
