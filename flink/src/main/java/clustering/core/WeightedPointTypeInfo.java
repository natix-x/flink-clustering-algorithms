package clustering.core;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.TypeSerializer;

/**
 * Custom {@link TypeInformation} for {@link WeightedPoint}.
 * Ensures Flink uses {@link WeightedPointSerializer} instead of falling back
 * to slower reflective POJO or Kryo serializers.
 */
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

    @Override
    public int getArity() {
        return 1; // Treated as a single opaque object
    }

    @Override
    public int getTotalFields() {
        return 1;
    }

    @Override
    public Class<WeightedPoint> getTypeClass() {
        return WeightedPoint.class;
    }

    @Override
    public boolean isKeyType() {
        return false; // Points are not used as keys; prevents unwanted shuffles
    }

    @Override
    public TypeSerializer<WeightedPoint> createSerializer(ExecutionConfig executionConfig) {
        return WeightedPointSerializer.INSTANCE;
    }

    @Override
    public String toString() {
        return "WeightedPoint";
    }

    @Override
    public boolean equals(Object otherObj) {
        return otherObj instanceof WeightedPointTypeInfo;
    }

    @Override
    public int hashCode() {
        return WeightedPointTypeInfo.class.hashCode();
    }

    @Override
    public boolean canEqual(Object otherObj) {
        return otherObj instanceof WeightedPointTypeInfo;
    }
}
