package clustering.benchmark.metrics;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Cumulative counters are differenced PER PROCESS, not on the pre-summed total.
 *
 *  <p>The distinction is not academic: summing across TaskManagers first and subtracting the
 *  sums second is correct only while the same set of processes is present at both reads. On a
 *  real 4-node run (29.08.2026) it was not, and the worker phases added up to 44.8 CPU-seconds
 *  against a run total of 31.7 — while the JobManager legs, being a single file, matched to the
 *  unit. These tests pin the fix, and pin it against the Spark port's rule
 *  ({@code ProcessCpuPlugin.sumDelta}), since both numbers land in the same column. */
class PhaseWindowTest {

    private static BenchmarkListener.Snapshot snap(long... cpuPerTaskManager) {
        BenchmarkListener.Snapshot s = new BenchmarkListener.Snapshot();
        for (int i = 0; i < cpuPerTaskManager.length; i++) {
            if (cpuPerTaskManager[i] >= 0) {
                s.cpuByProcess.put("host" + i + ".taskmanager.tm" + i + ".", cpuPerTaskManager[i]);
            }
        }
        s.recomputeTotals();
        return s;
    }

    @Test
    void deltaIsPerProcessAndSummedAfterwards() {
        BenchmarkListener.PhaseDelta d =
            BenchmarkListener.between(snap(100L, 80L), snap(250L, 180L));
        assertEquals((250 - 100) + (180 - 80), d.cpuNanos);
    }

    @Test
    void phasesTileTheRunWindowWhenEveryProcessIsPresentThroughout() {
        BenchmarkListener.Snapshot open = snap(100L, 100L);
        BenchmarkListener.Snapshot afterLoad = snap(250L, 200L);
        BenchmarkListener.Snapshot afterFit = snap(900L, 700L);
        BenchmarkListener.Snapshot end = snap(1000L, 800L);

        long phases = BenchmarkListener.between(open, afterLoad).cpuNanos
            + BenchmarkListener.between(afterLoad, afterFit).cpuNanos
            + BenchmarkListener.between(afterFit, end).cpuNanos;
        assertEquals(BenchmarkListener.between(open, end).cpuNanos, phases);
    }

    @Test
    void aProcessMissingFromOneReadDoesNotCorruptTheOthers() {
        // -1 = that TaskManager's file was not there at this read. Pre-summing would fold this
        // into one scalar and the subtraction would silently charge the gap to whichever phase
        // straddled it; per-process, tm0 is simply unaffected.
        BenchmarkListener.Snapshot from = snap(100L, -1L);
        BenchmarkListener.Snapshot to = snap(250L, 500L);

        // tm0 contributes its real delta; tm1, absent from `from`, contributes its whole reading
        // — the same rule the Spark port applies to a late-registering executor.
        assertEquals((250 - 100) + 500, BenchmarkListener.between(from, to).cpuNanos);
    }

    @Test
    void aProcessThatVanishedContributesNothingRatherThanANegative() {
        BenchmarkListener.Snapshot from = snap(100L, 400L);
        BenchmarkListener.Snapshot to = snap(250L, -1L);
        assertEquals(250 - 100, BenchmarkListener.between(from, to).cpuNanos);
    }

    @Test
    void aRestartedProcessFloorsAtZeroInsteadOfGoingNegative() {
        // Fresh JVM, counter below the earlier reading. On the pre-summed scalar this could be
        // masked by other processes' growth; per-process it is floored where it happens.
        BenchmarkListener.Snapshot from = snap(1000L, 100L);
        BenchmarkListener.Snapshot to = snap(40L, 300L);
        assertEquals(300 - 100, BenchmarkListener.between(from, to).cpuNanos);
    }

    @Test
    void scalarTotalsStayInStepWithTheMaps() {
        BenchmarkListener.Snapshot s = snap(100L, 250L);
        assertEquals(350L, s.executorCpuTimeNs);
    }
}
