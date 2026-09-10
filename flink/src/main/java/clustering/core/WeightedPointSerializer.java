package clustering.core;

import org.apache.flink.api.common.typeutils.SimpleTypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.base.TypeSerializerSingleton;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.linalg.DenseVector;

import java.io.IOException;

/**
 * Custom serializer for {@link WeightedPoint}.
 * Optimized for zero framing overhead and object reuse to minimize GC pressure.
 */
public final class WeightedPointSerializer extends TypeSerializerSingleton<WeightedPoint> {

    public static final WeightedPointSerializer INSTANCE = new WeightedPointSerializer();

    private static final long serialVersionUID = 1L;

    /**
     * Marker for null features to prevent silent deserialization into empty points.
     */
    private static final int NULL_FEATURE_MARKER = -1;

    private WeightedPointSerializer() {}

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public WeightedPoint createInstance() {
        return new WeightedPoint();
    }

    @Override
    public WeightedPoint copy(WeightedPoint source) {
        if (source == null) {
            return null;
        }

        DenseVector copiedFeatures = source.features == null
            ? null
            : new DenseVector(source.features.values.clone());

        return new WeightedPoint(copiedFeatures, source.weight);
    }

    @Override
    public WeightedPoint copy(WeightedPoint source, WeightedPoint reusedPoint) {
        if (source == null) {
            return null;
        }

        if (source.features == null) {
            reusedPoint.features = null;
        } else {
            double[] targetArray = getOrResizeCoordinateArray(reusedPoint, source.features.values.length);
            System.arraycopy(source.features.values, 0, targetArray, 0, targetArray.length);
        }
        reusedPoint.weight = source.weight;

        return reusedPoint;
    }

    @Override
    public int getLength() {
        return -1; // Variable length due to dynamic feature array
    }

    @Override
    public void serialize(WeightedPoint point, DataOutputView target) throws IOException {
        if (point.features == null) {
            target.writeInt(NULL_FEATURE_MARKER);
        } else {
            double[] coordinates = point.features.values;
            target.writeInt(coordinates.length);
            for (double coordinate : coordinates) {
                target.writeDouble(coordinate);
            }
        }
        target.writeDouble(point.weight);
    }

    @Override
    public WeightedPoint deserialize(DataInputView source) throws IOException {
        return deserialize(new WeightedPoint(), source);
    }

    @Override
    public WeightedPoint deserialize(WeightedPoint reusedPoint, DataInputView source) throws IOException {
        int featureCount = source.readInt();

        if (featureCount == NULL_FEATURE_MARKER) {
            reusedPoint.features = null;
        } else {
            double[] coordinates = getOrResizeCoordinateArray(reusedPoint, featureCount);
            for (int i = 0; i < featureCount; i++) {
                coordinates[i] = source.readDouble();
            }
        }
        reusedPoint.weight = source.readDouble();

        return reusedPoint;
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        int featureCount = source.readInt();
        target.writeInt(featureCount);

        // Include weight (1 double) in the copy count
        int elementsToCopy = featureCount == NULL_FEATURE_MARKER ? 1 : featureCount + 1;

        for (int i = 0; i < elementsToCopy; i++) {
            target.writeDouble(source.readDouble());
        }
    }

    @Override
    public TypeSerializerSnapshot<WeightedPoint> snapshotConfiguration() {
        return new WeightedPointSerializerSnapshot();
    }

    /**
     * Reuses the existing coordinate array or allocates a new one if the dimension changes.
     */
    private static double[] getOrResizeCoordinateArray(WeightedPoint reusedPoint, int requiredDimension) {
        if (reusedPoint.features == null || reusedPoint.features.values.length != requiredDimension) {
            reusedPoint.features = new DenseVector(new double[requiredDimension]);
        }
        return reusedPoint.features.values;
    }

    public static final class WeightedPointSerializerSnapshot extends SimpleTypeSerializerSnapshot<WeightedPoint> {
        public WeightedPointSerializerSnapshot() {
            super(() -> INSTANCE);
        }
    }
}
