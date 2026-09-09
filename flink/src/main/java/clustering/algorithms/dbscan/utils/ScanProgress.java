package clustering.algorithms.dbscan.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Time-throttled progress tracker for the O(m²) ε-graph scan.
 * Progress is measured in evaluated pairs rather than rows to accurately reflect the workload.
 */
public final class ScanProgress {

    private static final Logger logger = LoggerFactory.getLogger(ScanProgress.class);
    private static final long LOG_INTERVAL_NANOS = 30L * 1_000_000_000L;

    private final int totalRows;
    private final double totalPairs;
    private final long startTimeNanos;
    private long lastLogTimeNanos;

    public ScanProgress(int totalRows) {
        this.totalRows = totalRows;
        this.totalPairs = (double) totalRows * totalRows / 2;
        this.startTimeNanos = System.nanoTime();
        this.lastLogTimeNanos = startTimeNanos;
    }

    public void report(int processedRows, String executionContext) {
        long currentTimeNanos = System.nanoTime();

        if (currentTimeNanos - lastLogTimeNanos >= LOG_INTERVAL_NANOS) {
            lastLogTimeNanos = currentTimeNanos;
            double elapsedSeconds = (currentTimeNanos - startTimeNanos) / 1e9;
            double completionFraction = ((double) processedRows * totalRows - (double) processedRows * processedRows / 2) / totalPairs;
            double estimatedSecondsLeft = elapsedSeconds * (1 - completionFraction) / completionFraction;

            logger.info(String.format(
                "dbscanpp: step 3 ε-graph (%s) %.1f%% (%d/%d rows), %.0f s elapsed, %.0f s left",
                executionContext, completionFraction * 100, processedRows, totalRows, elapsedSeconds, estimatedSecondsLeft));
        }
    }
}
