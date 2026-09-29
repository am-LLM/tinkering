import unittest
from merkle_patricia_trie_proof import MerkleTree

class TestMerkle(unittest.TestCase):
    def test_root(self):
        t = MerkleTree(["tx1", "tx2", "tx3", "tx4"])
        self.assertEqual(len(t.get_root()), 64)
