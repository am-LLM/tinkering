import unittest
from aho_corasick_automaton import AhoCorasickAutomaton

class TestAhoCorasick(unittest.TestCase):
    def test_search(self):
        ac = AhoCorasickAutomaton(["he", "she", "his", "hers"])
        matches = ac.search_text("ushers")
        self.assertEqual(len(matches), 3)
