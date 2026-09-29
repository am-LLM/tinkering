import unittest
from disjoint_set_union_find import DisjointSetUnion

class TestDSU(unittest.TestCase):
    def test_dsu(self):
        dsu = DisjointSetUnion(5)
        self.assertTrue(dsu.union(0, 1))
        self.assertTrue(dsu.union(1, 2))
        self.assertFalse(dsu.union(0, 2))
        self.assertEqual(dsu.find(0), dsu.find(2))
