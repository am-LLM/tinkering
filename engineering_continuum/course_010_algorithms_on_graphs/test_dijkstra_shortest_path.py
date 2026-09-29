import unittest
from dijkstra_shortest_path_fibonacci import dijkstra_shortest_path

class TestDijkstra(unittest.TestCase):
    def test_shortest_path(self):
        g = {'A': {'B': 1, 'C': 4}, 'B': {'C': 2, 'D': 5}, 'C': {'D': 1}, 'D': {}}
        dist = dijkstra_shortest_path(g, 'A')
        self.assertEqual(dist['D'], 4)
