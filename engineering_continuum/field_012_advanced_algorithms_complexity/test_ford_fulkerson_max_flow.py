import unittest
from ford_fulkerson_max_flow import MaxFlowNetwork

class TestMaxFlow(unittest.TestCase):
    def test_edmonds_karp(self):
        net = MaxFlowNetwork(4)
        net.add_edge(0, 1, 10)
        net.add_edge(0, 2, 10)
        net.add_edge(1, 2, 2)
        net.add_edge(1, 3, 10)
        net.add_edge(2, 3, 10)
        self.assertEqual(net.edmonds_karp(0, 3), 20)
