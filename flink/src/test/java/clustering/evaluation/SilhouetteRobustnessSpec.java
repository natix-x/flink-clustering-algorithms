package clustering.evaluation;

import clustering.TestFixtures;
import clustering.core.EnvFactory;
import clustering.core.WeightedPoint;
import clustering.distance.EuclideanDistance;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.util.Collector;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** What the silhouette does when the input or the DRAW is degenerate — the cases where a wrong
 *  answer is a plausible-looking number rather than a crash, and therefore the ones the result file
 *  cannot be read without.
 *
 *  <p>Mirrors the Spark {@code SilhouetteRobustnessSpec} and {@code SilhouetteSubsampleSpec}. The
 *  scoring cases go through {@link SilhouetteEvaluator.Sample#score} directly, because a labelling
 *  the {@code (label, mass)} pair needs — a real cluster the SAMPLE under-drew — cannot be staged
 *  through a coordinate-keyed model; the draw cases go through {@link SilhouetteEvaluator.DrawMap},
 *  which is the whole of the sampling logic and needs no cluster to exercise. */
class SilhouetteRobustnessSpec {

    private static final double TOLERANCE = 1e-9;

    private static final EnvFactory ENVS = TestFixtures.localEnvs(2);

    // ------------------------------------------------------------- the scoring

    /** {@code trueMass <= 1} is a GENUINE singleton: one point with no neighbours, so {@code a} is
     *  undefined and Rousseeuw's convention is 0. */
    @Test
    void aGenuinelySingletonClusterStillScoresZero() {
        SilhouetteEvaluator.Sample sample = sample(
            new double[][] {{0.0}, {2.0}, {100.0}},
            new double[] {1.0, 1.0, 1.0},
            new int[] {0, 0, 1},
            new double[] {2.0, 1.0},      // sample mass
            new double[] {2.0, 1.0});     // FULL-data mass: cluster 1 really is a singleton

        assertEquals(0.0, sample.score(2, new double[2], EuclideanDistance.INSTANCE));
    }

    /** A REAL cluster the sample drew one member of is a different case, and scoring it 0 would be a
     *  second downward bias — discrete and k-dependent — on top of the min-of-noisy-b one. NaN, so
     *  the caller excludes it from the mean and counts it. */
    @Test
    void aClusterTheSampleUnderDrewIsExcludedNotScoredZero() {
        SilhouetteEvaluator.Sample sample = sample(
            new double[][] {{0.0}, {2.0}, {100.0}},
            new double[] {1.0, 1.0, 1.0},
            new int[] {0, 0, 1},
            new double[] {2.0, 1.0},         // the sample drew ONE member of cluster 1
            new double[] {2.0, 5000.0});     // the full data has 5000 of them

        assertTrue(Double.isNaN(sample.score(2, new double[2], EuclideanDistance.INSTANCE)),
            "an under-drawn cluster must be unscorable, not scored 0");
    }

    /** Without a {@code Population} the evaluator only has the sample's own masses, so it cannot
     *  tell the two cases apart and falls back to the singleton rule. Stated as a test so the
     *  fallback is a documented behaviour rather than an accident. */
    @Test
    void withoutAPopulationTheUnderDrawnClusterFallsBackToTheSingletonRule() {
        double[] sampleMass = {2.0, 1.0};
        SilhouetteEvaluator.Sample sample = sample(
            new double[][] {{0.0}, {2.0}, {100.0}},
            new double[] {1.0, 1.0, 1.0},
            new int[] {0, 0, 1},
            sampleMass,
            sampleMass);                     // trueMass == slotMass: no population was supplied

        assertEquals(0.0, sample.score(2, new double[2], EuclideanDistance.INSTANCE));
    }

    /** {@code min(1, w)} rather than a flat 1: for a FRACTIONAL weight "one copy" does not exist,
     *  and subtracting a whole one gives a denominator smaller than the neighbours' mass. Two rows
     *  of weight 0.6 at distance 1 must give {@code a = 1}, not the {@code 0.6/0.2 = 3} a flat 1
     *  produces. */
    @Test
    void aFractionalWeightDegradesToTheLeaveThisRowOutMean() {
        SilhouetteEvaluator.Sample sample = sample(
            new double[][] {{0.0}, {1.0}, {100.0}, {101.0}},
            new double[] {0.6, 0.6, 0.6, 0.6},
            new int[] {0, 0, 1, 1},
            new double[] {1.2, 1.2},
            new double[] {1.2, 1.2});

        // a = 1 (the single other member at distance 1), b = (100 + 101)/2 = 100.5 -> s = 99.5/100.5
        assertEquals(99.5 / 100.5, sample.score(0, new double[2], EuclideanDistance.INSTANCE), TOLERANCE);
    }

    /** A non-finite coordinate is dropped at collection time and COUNTED, so it costs its own row
     *  rather than every other point's {@code b}. Left in, its cluster's mean distance would be NaN,
     *  NaN never wins a {@code min}, and every point of the neighbouring cluster would silently take
     *  its {@code b} from some far-away cluster — the score inflates toward 1 with no symptom. */
    @Test
    void aNonFiniteCoordinateCostsItsOwnRowNotEveryOtherPointsB() {
        List<SilhouetteEvaluator.Drawn> drawn = Arrays.asList(
            new SilhouetteEvaluator.Drawn(0, new double[] {0.0}, 1.0),
            new SilhouetteEvaluator.Drawn(0, new double[] {2.0}, 1.0),
            new SilhouetteEvaluator.Drawn(1, new double[] {Double.NaN}, 1.0),
            new SilhouetteEvaluator.Drawn(1, new double[] {Double.POSITIVE_INFINITY}, 1.0),
            new SilhouetteEvaluator.Drawn(1, new double[] {10.0}, 1.0),
            new SilhouetteEvaluator.Drawn(1, new double[] {12.0}, 1.0));

        SilhouetteEvaluator.Collected collected = SilhouetteEvaluator.collectFrom(drawn, null);

        assertEquals(2, collected.droppedRows, "both non-finite rows must be refused");
        assertEquals(4, collected.sample.size());
        // The survivors are exactly the SilhouetteSpec fixture, so the score is unchanged.
        double[] scores = new double[4];
        for (int i = 0; i < 4; i++) {
            scores[i] = collected.sample.score(i, new double[2], EuclideanDistance.INSTANCE);
        }
        assertEquals(79.0 / 99.0, (scores[0] + scores[1] + scores[2] + scores[3]) / 4.0, TOLERANCE);
    }

    /** Corrupt weights are neutralised in the draw, so they never reach a Σ. Asserted through
     *  {@code collectFrom} as well, which is the belt-and-braces guard for a direct caller. */
    @Test
    void aCorruptWeightCostsItsOwnRow() {
        List<SilhouetteEvaluator.Drawn> drawn = Arrays.asList(
            new SilhouetteEvaluator.Drawn(0, new double[] {0.0}, 1.0),
            new SilhouetteEvaluator.Drawn(0, new double[] {2.0}, Double.NaN),
            new SilhouetteEvaluator.Drawn(1, new double[] {10.0}, -3.0),
            new SilhouetteEvaluator.Drawn(1, new double[] {12.0}, Double.POSITIVE_INFINITY));

        SilhouetteEvaluator.Collected collected = SilhouetteEvaluator.collectFrom(drawn, null);

        assertEquals(3, collected.droppedRows);
        assertEquals(1, collected.sample.size());
    }

    /** The sample layout must be a pure function of the drawn SET: slots ascending by label, rows
     *  sorted, so no floating-point summation order can travel from "which subtask finished first"
     *  into the reported score. */
    @Test
    void theSampleLayoutDoesNotDependOnCollectionOrder() {
        List<SilhouetteEvaluator.Drawn> forward = Arrays.asList(
            new SilhouetteEvaluator.Drawn(1, new double[] {10.0}, 1.0),
            new SilhouetteEvaluator.Drawn(0, new double[] {2.0}, 1.0),
            new SilhouetteEvaluator.Drawn(1, new double[] {12.0}, 1.0),
            new SilhouetteEvaluator.Drawn(0, new double[] {0.0}, 1.0));
        List<SilhouetteEvaluator.Drawn> reversed = new ArrayList<>(forward);
        java.util.Collections.reverse(reversed);

        SilhouetteEvaluator.Sample a = SilhouetteEvaluator.collectFrom(forward, null).sample;
        SilhouetteEvaluator.Sample b = SilhouetteEvaluator.collectFrom(reversed, null).sample;

        assertArrayEqualsDeep(a.points, b.points);
        assertArrayEquals(a.slots, b.slots);
        assertArrayEquals(a.weights, b.weights);
    }

    // ---------------------------------------------------------------- the draw

    /** At unit weights the mass-proportional rule IS the Bernoulli filter: {@code L < 1}, so a row
     *  is drawn at most once, at the target rate. This is why Flink needs one branch where Spark
     *  needs two. */
    @Test
    void unweightedInputKeepsEachRowAtMostOnceAtTheTargetRate() {
        int rows = 4000;
        int sampleSize = 400;
        List<SilhouetteEvaluator.Drawn> drawn = drawAll(rows, 1.0, sampleSize / (double) rows, 42L, sampleSize);

        Set<String> distinct = new HashSet<>();
        for (SilhouetteEvaluator.Drawn row : drawn) {
            assertTrue(distinct.add(Arrays.toString(row.coordinates)),
                "an unweighted row must not be drawn twice");
            assertEquals(1.0, row.weight);
        }
        // Binomial(4000, 0.1): sd = 9.5, so +-5 sd is a 48-wide band.
        assertTrue(Math.abs(drawn.size() - sampleSize) < 50,
            "drew " + drawn.size() + ", expected about " + sampleSize);
    }

    /** A heavy row gets a share of the sample proportional to its MASS, and every draw comes back
     *  with weight 1.0 — sampling proportionally AND keeping the original weight would count the
     *  mass twice. */
    @Test
    void aHeavyRowGetsAShareOfTheSampleProportionalToItsMass() {
        // One row of mass 900 against 100 of mass 1: totalMass 1000, sampleSize 100 -> lambda = 0.1.
        List<SilhouetteEvaluator.Drawn> heavy = drawOne(new double[] {0.0}, 900.0, 0.1, 42L, 100);

        assertEquals(90, heavy.size(), "floor(900 * 0.1) = 90 copies go in deterministically");
        for (SilhouetteEvaluator.Drawn row : heavy) {
            assertEquals(1.0, row.weight, "every draw stands for exactly one point");
        }
    }

    /** One row must never be able to become the whole sample, whatever weight reaches the draw. */
    @Test
    void theCopyCapBoundsASingleRowsContribution() {
        assertEquals(10, drawOne(new double[] {0.0}, 1e9, 0.1, 42L, 10).size());
    }

    /** Noise and corrupt weights are outside the draw entirely. */
    @Test
    void noiseAndCorruptWeightsNeverEnterTheDraw() {
        assertTrue(drawOne(new double[] {0.0}, 1.0, Double.NaN, 42L, 100, -1).isEmpty(),
            "a noise point must not be drawn");
        assertTrue(drawOne(new double[] {0.0}, -1.0, Double.NaN, 42L, 100).isEmpty(),
            "a negative weight must not be drawn");
        assertTrue(drawOne(new double[] {0.0}, Double.NaN, Double.NaN, 42L, 100).isEmpty(),
            "a NaN weight must not be drawn");
    }

    /** The draw is a pure function of (row content, seed) — never of the record-to-subtask
     *  assignment, which is what the benchmark matrix varies. A per-subtask RNG would make the
     *  reported silhouette drift with worker count on a bit-identical clustering, drift
     *  indistinguishable from the instability the thesis MEASURES. */
    @Test
    void theDrawIsAPureFunctionOfContentAndSeed() {
        double u = SilhouetteEvaluator.rowUniform(new double[] {1.0, 2.0}, 3.0, 42L);

        assertEquals(u, SilhouetteEvaluator.rowUniform(new double[] {1.0, 2.0}, 3.0, 42L));
        assertNotEquals(u, SilhouetteEvaluator.rowUniform(new double[] {1.0, 2.0}, 3.0, 43L));
        assertNotEquals(u, SilhouetteEvaluator.rowUniform(new double[] {1.0, 2.5}, 3.0, 42L));
        assertNotEquals(u, SilhouetteEvaluator.rowUniform(new double[] {1.0, 2.0}, 4.0, 42L));
        assertTrue(u >= 0.0 && u < 1.0, "the draw must land in [0, 1), got " + u);
    }

    /** ...and it is uniform enough to be a draw at all: adjacent coordinates must decorrelate, or a
     *  grid-snapped dataset (NYC TLC) would sample a stripe of itself. */
    @Test
    void theDrawIsUniformOverAGrid() {
        int inFirstDecile = 0;
        for (int i = 0; i < 10_000; i++) {
            if (SilhouetteEvaluator.rowUniform(new double[] {i, i + 1}, 1.0, 7L) < 0.1) {
                inFirstDecile++;
            }
        }
        // Binomial(10000, 0.1): sd = 30, so +-5 sd is a 150-wide band around 1000.
        assertTrue(Math.abs(inFirstDecile - 1000) < 150, "first decile held " + inFirstDecile);
    }

    // ------------------------------------------------------------- end to end

    /** A subsampled run reports what it scored: the achieved size is Binomial around
     *  {@code sampleSize}, and the score has to be read next to that number rather than assumed to
     *  be over exactly {@code sampleSize} points. */
    @Test
    void aSubsampledRunReportsWhatItScored() {
        int perCluster = 500;
        double[][] points = twoBlobs(perCluster);
        int[] labels = new int[points.length];
        for (int i = perCluster; i < points.length; i++) {
            labels[i] = 1;
        }
        LabelByCoordinateModel model = new LabelByCoordinateModel(points, labels);

        SilhouetteEvaluator.Outcome outcome = new SilhouetteEvaluator(EuclideanDistance.INSTANCE)
            .measure(model, TestFixtures.source(asRows(points)), ENVS, 2, 200, 42L, population(points, labels));

        assertEquals(2, outcome.sampleClusters);
        assertTrue(Math.abs(outcome.scoredPoints - 200) < 60,
            "scored " + outcome.scoredPoints + ", expected about 200");
        assertEquals(0, outcome.unscoredPoints);
        assertTrue(outcome.score > 0.9, "two well-separated blobs should score high, got " + outcome.score);
    }

    /** Same clustering, same seed, different PARALLELISM: the reported score must be bit-identical.
     *  This is the property the content hash and the collection-order sort exist for, and the one a
     *  per-subtask RNG breaks silently. */
    @Test
    void theScoreDoesNotMoveWithParallelism() {
        int perCluster = 300;
        double[][] points = twoBlobs(perCluster);
        int[] labels = new int[points.length];
        for (int i = perCluster; i < points.length; i++) {
            labels[i] = 1;
        }
        LabelByCoordinateModel model = new LabelByCoordinateModel(points, labels);
        SilhouetteEvaluator evaluator = new SilhouetteEvaluator(EuclideanDistance.INSTANCE);
        SilhouetteEvaluator.Population population = population(points, labels);

        SilhouetteEvaluator.Outcome one = evaluator.measure(
            model, TestFixtures.source(asRows(points)), TestFixtures.localEnvs(1), 1, 150, 42L, population);
        SilhouetteEvaluator.Outcome four = evaluator.measure(
            model, TestFixtures.source(asRows(points)), TestFixtures.localEnvs(4), 4, 150, 42L, population);

        assertEquals(one.scoredPoints, four.scoredPoints);
        assertEquals(one.score, four.score, 0.0, "the score must not depend on parallelism");
    }

    /** An input lighter than the budget skips the draw and is scored exactly. */
    @Test
    void anInputLighterThanTheBudgetIsScoredExactly() {
        double[][] points = {{0.0}, {2.0}, {10.0}, {12.0}};
        int[] labels = {0, 0, 1, 1};
        LabelByCoordinateModel model = new LabelByCoordinateModel(points, labels);

        SilhouetteEvaluator.Outcome outcome = new SilhouetteEvaluator(EuclideanDistance.INSTANCE)
            .measure(model, TestFixtures.source(asRows(points)), ENVS, 2, 1000, 42L, population(points, labels));

        assertEquals(4, outcome.scoredPoints);
        assertEquals(79.0 / 99.0, outcome.score, TOLERANCE);
    }

    /** Fewer than two clusters: 0.0, and the counters say why rather than leaving the reader to
     *  guess whether the labelling or the draw collapsed. */
    @Test
    void fewerThanTwoClustersYieldsZeroAndSaysSo() {
        double[][] points = {{0.0}, {2.0}, {9.0}};
        int[] labels = {0, 0, 0};

        SilhouetteEvaluator.Outcome outcome = new SilhouetteEvaluator(EuclideanDistance.INSTANCE)
            .measure(new LabelByCoordinateModel(points, labels),
                TestFixtures.source(asRows(points)), ENVS, 2, null, 0L, null);

        assertEquals(0.0, outcome.score);
        assertEquals(0, outcome.scoredPoints);
        assertEquals(1, outcome.sampleClusters);
        assertEquals(3, outcome.unscoredPoints);
    }

    // ------------------------------------------------------------ the path plan

    /** The m² scoring is only worth a Flink job when the driver would be busy long enough, the
     *  cluster has materially more parallelism than the driver has cores, and the sample fits in the
     *  JobGraph. At the default {@code sampleSize} the first condition alone keeps it local, which is
     *  the honest answer for Flink and the reason this decides where Spark always distributes. */
    @Test
    void theDefaultSampleSizeScoresOnTheDriver() {
        assertNotNull(SilhouetteEvaluator.planFor(10_000, 1024, 1000, 48, 10_000L * 1024 * 8),
            "10 000 points is seconds of work; a job would be pure overhead");
    }

    @Test
    void aLargeSampleOnAMuchBiggerClusterIsDistributed() {
        assertNull(SilhouetteEvaluator.planFor(800_000, 8, 1000, 48, 800_000L * 8 * 8));
    }

    @Test
    void aSingleNodeRunIsNeverDistributed() {
        String reason = SilhouetteEvaluator.planFor(800_000, 8, 12, 12, 800_000L * 8 * 8);
        assertNotNull(reason);
        assertTrue(reason.contains("parallelism"), reason);
    }

    @Test
    void aSampleTooBigForTheJobGraphIsRefusedTheCluster() {
        String reason = SilhouetteEvaluator.planFor(200_000, 1024, 1000, 48, 200_000L * 1024 * 8);
        assertNotNull(reason);
        assertTrue(reason.contains("shipping"), reason);
    }

    /** The two scoring paths must agree bit for bit. They run the same kernel and both fold in
     *  ascending index order over a sample that was sorted at collection time, so this is structural
     *  rather than a tolerance — but the distributed path is the one the plan keeps switched off at
     *  every size the rest of the suite exercises, so it would otherwise ship untested (and a POJO
     *  that silently fell back to Kryo, or a closure that failed to serialise, would only surface on
     *  a large production run). */
    @Test
    void theDistributedAndDriverLocalScoringPathsAgree() {
        int perCluster = 60;
        double[][] points = twoBlobs(perCluster);
        List<SilhouetteEvaluator.Drawn> drawn = new ArrayList<>();
        for (int i = 0; i < points.length; i++) {
            drawn.add(new SilhouetteEvaluator.Drawn(i < perCluster ? 0 : 1, points[i], 1.0));
        }
        SilhouetteEvaluator.Sample sample = SilhouetteEvaluator.collectFrom(drawn, null).sample;
        SilhouetteEvaluator evaluator = new SilhouetteEvaluator(EuclideanDistance.INSTANCE);

        assertArrayEquals(
            evaluator.scoreLocally(sample),
            evaluator.scoreDistributed(sample, TestFixtures.localEnvs(3), 3));
    }

    // ------------------------------------------------------------------ helpers

    private static SilhouetteEvaluator.Sample sample(double[][] points, double[] weights, int[] slots,
                                                     double[] slotMass, double[] trueMass) {
        return new SilhouetteEvaluator.Sample(points, weights, slots, slotMass, trueMass);
    }

    /** {@code perMassLambda = sampleSize / totalMass}; NaN means "emit every labelled point". */
    private static List<SilhouetteEvaluator.Drawn> drawOne(
            double[] coordinates, double weight, double perMassLambda, long seed, int cap) {
        return drawOne(coordinates, weight, perMassLambda, seed, cap, 0);
    }

    private static List<SilhouetteEvaluator.Drawn> drawOne(
            double[] coordinates, double weight, double perMassLambda, long seed, int cap, int label) {
        List<SilhouetteEvaluator.Drawn> out = new ArrayList<>();
        new SilhouetteEvaluator.DrawMap(features -> label, perMassLambda, seed, cap)
            .flatMap(new WeightedPoint(new DenseVector(coordinates), weight), collectorInto(out));
        return out;
    }

    private static List<SilhouetteEvaluator.Drawn> drawAll(
            int rows, double weight, double perMassLambda, long seed, int cap) {
        List<SilhouetteEvaluator.Drawn> out = new ArrayList<>();
        SilhouetteEvaluator.DrawMap draw =
            new SilhouetteEvaluator.DrawMap(features -> 0, perMassLambda, seed, cap);
        Collector<SilhouetteEvaluator.Drawn> collector = collectorInto(out);
        for (int i = 0; i < rows; i++) {
            draw.flatMap(new WeightedPoint(new DenseVector(new double[] {i}), weight), collector);
        }
        return out;
    }

    private static Collector<SilhouetteEvaluator.Drawn> collectorInto(List<SilhouetteEvaluator.Drawn> out) {
        return new Collector<SilhouetteEvaluator.Drawn>() {
            @Override public void collect(SilhouetteEvaluator.Drawn record) {
                out.add(record);
            }

            @Override public void close() {}
        };
    }

    /** Two tight, well-separated blobs of {@code perCluster} distinct points each. */
    private static double[][] twoBlobs(int perCluster) {
        double[][] points = new double[2 * perCluster][];
        for (int i = 0; i < perCluster; i++) {
            points[i] = new double[] {i * 0.001, 0.0};
            points[perCluster + i] = new double[] {100.0 + i * 0.001, 0.0};
        }
        return points;
    }

    private static List<double[]> asRows(double[][] points) {
        List<double[]> out = new ArrayList<>(points.length);
        for (double[] point : points) {
            out.add(point.clone());
        }
        return out;
    }

    /** Full-data per-cluster mass, which on unweighted input is the row count. */
    private static SilhouetteEvaluator.Population population(double[][] points, int[] labels) {
        Map<Integer, Double> mass = new HashMap<>();
        for (int label : labels) {
            if (label >= 0) {
                mass.merge(label, 1.0, Double::sum);
            }
        }
        return new SilhouetteEvaluator.Population(mass);
    }

    private static void assertArrayEquals(int[] expected, int[] actual) {
        assertEquals(Arrays.toString(expected), Arrays.toString(actual));
    }

    private static void assertArrayEquals(double[] expected, double[] actual) {
        assertEquals(Arrays.toString(expected), Arrays.toString(actual));
    }

    private static void assertArrayEqualsDeep(double[][] expected, double[][] actual) {
        assertEquals(Arrays.deepToString(expected), Arrays.deepToString(actual));
    }
}
