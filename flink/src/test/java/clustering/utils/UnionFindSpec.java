package clustering.utils;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Union-find, in particular the min-index rule the reproducible cluster ids depend on. */
class UnionFindSpec {

    @Test
    void findReturnsTheSmallestIndexInTheComponent() {
        UnionFind uf = new UnionFind(6);
        uf.union(4, 2);
        uf.union(2, 5);
        assertEquals(2, uf.find(4));
        assertEquals(2, uf.find(5));
        assertEquals(2, uf.find(2));
    }

    @Test
    void componentIdsAreContiguousAndOrderedByMinimumMember() {
        UnionFind uf = new UnionFind(7);
        uf.union(5, 6);   // component {5,6}
        uf.union(1, 3);   // component {1,3}
        // Singletons: 0, 2, 4
        assertArrayEquals(new int[] {0, 1, 2, 1, 3, 4, 4}, uf.componentIds());
    }

    @Test
    void unionIsIdempotentAndOrderIndependent() {
        UnionFind a = new UnionFind(4);
        a.union(0, 3);
        a.union(3, 0);
        a.union(1, 2);

        UnionFind b = new UnionFind(4);
        b.union(3, 0);
        b.union(2, 1);

        assertArrayEquals(a.componentIds(), b.componentIds());
    }
}
