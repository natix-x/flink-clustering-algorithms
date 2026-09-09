package clustering.core;

import org.apache.flink.api.common.typeutils.SimpleTypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.base.TypeSerializerSingleton;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.linalg.DenseVector;
import java.io.IOException;

/** Wire format for {@link WeightedPoint}: {@code int dimension}, then that many doubles, then the
 *  weight. Hand-written rather than left to Flink's {@code PojoSerializer} for two reasons that
 *  both show up per record, on every point of a 1.4-billion-row set:
 *
 *  <ul>
 *    <li><b>No framing.</b> The POJO serializer writes a null-flag byte per nullable field and
 *        reaches fields reflectively. Here the overhead over a bare {@code DenseVector} is exactly
 *        the 8 bytes of the weight — nothing else.</li>
 *    <li><b>Reuse.</b> {@link #deserialize(WeightedPoint, DataInputView)} fills the record and its
 *        coordinate array in place when the dimension matches, so a fold over cached points
 *        allocates one record per ITERATOR rather than one per point. That is what keeps the extra
 *        record wrapper off the GC's back in the iterative algorithms, whose whole design is to
 *        walk the cache once per round.</li>
 *  </ul>
 *
 *  Stateless, hence a singleton: {@link TypeSerializerSingleton} makes {@code duplicate()} return
 *  {@code this} and defines equality by class. */
public final class WeightedPointSerializer extends TypeSerializerSingleton<WeightedPoint> {

    public static final WeightedPointSerializer INSTANCE = new WeightedPointSerializer();

    private static final long serialVersionUID = 1L;

    /** Marks an absent feature vector, so a malformed record fails on read rather than silently
     *  deserialising as an empty point. */
    private static final int NULL_FEATURES = -1;

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
    public WeightedPoint copy(WeightedPoint from) {
        if (from == null) {
            return null;
        }
        DenseVector features =
            from.features == null ? null : new DenseVector(from.features.values.clone());
        return new WeightedPoint(features, from.weight);
    }

    @Override
    public WeightedPoint copy(WeightedPoint from, WeightedPoint reuse) {
        if (from == null) {
            return null;
        }
        if (from.features == null) {
            reuse.features = null;
        } else {
            double[] target = coordinateArray(reuse, from.features.values.length);
            System.arraycopy(from.features.values, 0, target, 0, target.length);
        }
        reuse.weight = from.weight;
        return reuse;
    }

    /** Variable length: the dimension is per record (it is constant within a dataset, but the
     *  serializer is not told that). */
    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(WeightedPoint record, DataOutputView target) throws IOException {
        if (record.features == null) {
            target.writeInt(NULL_FEATURES);
        } else {
            double[] values = record.features.values;
            target.writeInt(values.length);
            for (double value : values) {
                target.writeDouble(value);
            }
        }
        target.writeDouble(record.weight);
    }

    @Override
    public WeightedPoint deserialize(DataInputView source) throws IOException {
        return deserialize(new WeightedPoint(), source);
    }

    @Override
    public WeightedPoint deserialize(WeightedPoint reuse, DataInputView source) throws IOException {
        int dimension = source.readInt();
        if (dimension == NULL_FEATURES) {
            reuse.features = null;
        } else {
            double[] values = coordinateArray(reuse, dimension);
            for (int i = 0; i < dimension; i++) {
                values[i] = source.readDouble();
            }
        }
        reuse.weight = source.readDouble();
        return reuse;
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        int dimension = source.readInt();
        target.writeInt(dimension);
        int doubles = dimension == NULL_FEATURES ? 1 : dimension + 1;
        for (int i = 0; i < doubles; i++) {
            target.writeDouble(source.readDouble());
        }
    }

    @Override
    public TypeSerializerSnapshot<WeightedPoint> snapshotConfiguration() {
        return new WeightedPointSerializerSnapshot();
    }

    /** The reusable record's coordinate array, resized only when the dimension actually changes. */
    private static double[] coordinateArray(WeightedPoint reuse, int dimension) {
        if (reuse.features == null || reuse.features.values.length != dimension) {
            reuse.features = new DenseVector(new double[dimension]);
        }
        return reuse.features.values;
    }

    /** Stateless format, so the snapshot carries nothing but the serializer's identity. */
    public static final class WeightedPointSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<WeightedPoint> {

        public WeightedPointSerializerSnapshot() {
            super(() -> INSTANCE);
        }
    }
}
