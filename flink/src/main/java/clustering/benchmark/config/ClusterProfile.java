package clustering.benchmark.config;


public abstract class ClusterProfile {

    public abstract String name();

    /** true -> create a local Flink environment; false -> use the cluster context env. */
    public abstract boolean isLocal();

    public static ClusterProfile fromName(String name) {
        switch (name.toLowerCase()) {
            case "local": return new LocalProfile();
            case "ares":  return new AresProfile();
            default: throw new IllegalArgumentException("Unknown cluster profile: " + name);
        }
    }

    public static final class LocalProfile extends ClusterProfile {
        @Override public String name() { return "local"; }
        @Override public boolean isLocal() { return true; }
    }

    public static final class AresProfile extends ClusterProfile {
        @Override public String name() { return "ares"; }
        @Override public boolean isLocal() { return false; }
    }
}
