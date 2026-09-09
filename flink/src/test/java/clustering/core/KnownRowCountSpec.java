package clustering.core;

import clustering.TestFixtures;
import clustering.algorithms.kmedoids.hybrid.CLARA;
import clustering.distance.EuclideanDistance;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** A row count the benchmark job already paid for must not be paid for again.
 *
 *  <p>On Flink a count is a whole JOB that re-reads the source — there is no cross-job cache — so
 *  an algorithm calling {@code Datasets.count} after the `load` phase already counted was a second
 *  full pass over the dataset per run, to learn a number the job had in a local variable. The fix
 *  travels with the source rather than through the {@code Clusterer} seam, so no algorithm had to
 *  change and none can forget to use it.
 *
 *  <p>The assertions below are about JOB SUBMISSIONS, not timings: the counting source records how
 *  many times its stream was built, which is exactly what a redundant count costs. */
class KnownRowCountSpec {

    private static List<double[]> rows(int n) {
        List<double[]> rows = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            rows.add(new double[] {i % 10, (i % 7) * 1.0});
        }
        return rows;
    }

    /** Wraps a source and counts how often a job was built from it. */
    private static final class CountingSource implements PointSource {
        private final PointSource delegate;
        final AtomicInteger streamsBuilt = new AtomicInteger();

        CountingSource(PointSource delegate) {
            this.delegate = delegate;
        }

        @Override
        public DataStream<WeightedPoint> create(StreamExecutionEnvironment env) {
            streamsBuilt.incrementAndGet();
            return delegate.create(env);
        }
    }

    @Test
    void aSourceWithoutAKnownCountStillRunsTheJob() {
        CountingSource source = new CountingSource(TestFixtures.source(rows(500)));
        assertFalse(source.knownRowCount().isPresent(), "a plain source knows nothing");

        assertEquals(500L, Datasets.count(source, TestFixtures.localEnvs(2)));
        assertEquals(1, source.streamsBuilt.get(), "counting an unknown source costs one job");
    }

    @Test
    void aKnownCountIsAnsweredWithoutBuildingAJob() {
        CountingSource underlying = new CountingSource(TestFixtures.source(rows(500)));
        PointSource counted = PointSource.withKnownRowCount(underlying, 500L);

        assertEquals(500L, Datasets.count(counted, TestFixtures.localEnvs(2)));
        assertEquals(0, underlying.streamsBuilt.get(),
            "a known count must not submit a job — that is the whole point");
        // and it still produces the data when actually asked for it
        assertEquals(500, Datasets.collectAll(counted, TestFixtures.localEnvs(2)).size());
        assertEquals(1, underlying.streamsBuilt.get());
    }

    /** The safety property: a derived source is a different instance, so it reports NO count and
     *  can never inherit one describing rows it does not have. */
    @Test
    void aDerivedSourceDoesNotInheritTheCount() {
        PointSource counted = PointSource.withKnownRowCount(TestFixtures.source(rows(500)), 500L);
        PointSource filtered = env -> counted.create(env).filter(p -> p.features.values[0] < 5.0);

        assertEquals(OptionalLong.empty(), filtered.knownRowCount());
        assertTrue(Datasets.count(filtered, TestFixtures.localEnvs(2)) < 500L,
            "the derived source must be counted for real, not handed its parent's number");
    }

    /** End of the chain: an algorithm that needs n gets it from the source, so it submits no count
     *  job of its own. `clara` is the case that matters — `dbscanpp` takes the identical route. */
    @Test
    void claraDoesNotRecountWhenTheJobAlreadyDid() {
        CountingSource underlying = new CountingSource(TestFixtures.source(rows(2000)));
        PointSource counted = PointSource.withKnownRowCount(underlying, 2000L);

        new CLARA(3, 2, 200, 20, EuclideanDistance.INSTANCE, "fastpam", 42L)
            .fit(counted, TestFixtures.localEnvs(2), 2);

        // Without the known count this run built one extra stream for its own count job.
        assertEquals(1, underlying.streamsBuilt.get(),
            "clara should build exactly the iteration job's stream, having been told n");
    }
}
