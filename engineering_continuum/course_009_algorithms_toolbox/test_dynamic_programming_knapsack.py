import unittest
from dynamic_programming_knapsack import solve_knapsack_dp

class TestKnapsack(unittest.TestCase):
    def test_dp(self):
        val = solve_knapsack_dp([2, 3, 4], [3, 4, 5], 5)
        self.assertEqual(val, 7)
