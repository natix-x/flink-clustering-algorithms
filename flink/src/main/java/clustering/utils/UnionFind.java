package clustering.utils;

import java.util.Arrays;

/** Driver-local union-find (disjoint sets) over {@code size} elements. Java mirror of the
 *  Spark {@code clustering.utils.UnionFind}.
 *
 *  {@code union} attaches the larger root under the smaller (no union-by-rank), so
 *  {@code find(i)} returns the smallest index in i's component — which makes derived
 *  cluster ids reproducible. */
public final class UnionFind {

    private final int[] parent;

    public UnionFind(int size) {
        parent = new int[size];
        for (int i = 0; i < size; i++) {
            parent[i] = i;
        }
    }

    /** Representative of {@code i}'s component: the smallest index in it. */
    public int find(int i) {
        int root = i;
        while (parent[root] != root) {
            root = parent[root];
        }
        // Path compression, iterative.
        int node = i;
        while (parent[node] != root) {
            int next = parent[node];
            parent[node] = root;
            node = next;
        }
        return root;
    }

    public void union(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra != rb) {
            if (ra < rb) {
                parent[rb] = ra;
            } else {
                parent[ra] = rb;
            }
        }
    }

    /** Contiguous component ids from 0, numbered by ascending minimum member index — so
     *  element {@code 0}'s component is always id 0. */
    public int[] componentIds() {
        int[] roots = new int[parent.length];
        for (int i = 0; i < parent.length; i++) {
            roots[i] = find(i);
        }
        int[] distinctRoots = Arrays.stream(roots).distinct().sorted().toArray();
        int[] idOfRoot = new int[parent.length];
        for (int id = 0; id < distinctRoots.length; id++) {
            idOfRoot[distinctRoots[id]] = id;
        }
        int[] out = new int[parent.length];
        for (int i = 0; i < roots.length; i++) {
            out[i] = idOfRoot[roots[i]];
        }
        return out;
    }
}
