package clustering.benchmark.config;

/** Execution environment for a run. Mirrors the Spark repo's sealed
 *  {@code ClusterProfile}: local laptop vs Cyfronet Ares. {@code isLocal()}
 *  drives whether Flink uses a local MiniCluster env or the submitted-context env. */
public abstract class ClusterProfile {

    public abstract String name();

    public abstract String outputDir();

    /** true -> create a local Flink environment; false -> use the cluster context env. */
    public abstract boolean isLocal();

    public static ClusterProfile fromName(String name) {
        switch (name.toLowerCase()) {
            case "local": return new LocalProfile();
            case "ares":  return new AresProfile();
            default: throw new IllegalArgumentException("Unknown cluster profile: " + name);
        }
    }

    /** CLUSTERING_PROFILE wins, then SLURM detection, else local. */
    public static ClusterProfile fromEnv() {
        String explicit = System.getenv("CLUSTERING_PROFILE");
        if (explicit != null && !explicit.isEmpty()) {
            return fromName(explicit);
        }
        return System.getenv("SLURM_JOB_ID") != null ? new AresProfile() : new LocalProfile();
    }

    public static final class LocalProfile extends ClusterProfile {
        @Override public String name() { return "local"; }
        @Override public boolean isLocal() { return true; }
        @Override public String outputDir() {
            return System.getProperty("user.dir", ".") + "/benchmark-results";
        }
    }

    public static final class AresProfile extends ClusterProfile {
        @Override public String name() { return "ares"; }
        @Override public boolean isLocal() { return false; }
        @Override public String outputDir() {
            String scratch = System.getenv().getOrDefault("SCRATCH", ".");
            return scratch + "/clustering-runs";
        }
    }
}