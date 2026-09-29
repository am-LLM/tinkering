import unittest
from overlap_layout_consensus_assembler import OLCAssembler

class TestOLC(unittest.TestCase):
    def test_assemble(self):
        reads = ["ATTGC", "TGCGA", "CGATA"]
        olc = OLCAssembler(reads, min_overlap=3)
        res = olc.greedy_assemble()
        self.assertEqual(res, "ATTGCGATA")
