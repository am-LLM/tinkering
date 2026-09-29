import unittest
from drake_equation_extremophile_model import DrakeExtremophileModel

class TestAstrobiology(unittest.TestCase):
    def test_signatures(self):
        m = DrakeExtremophileModel(1.5, 0.8, 0.4)
        n = m.estimate_active_biosignatures(0.3, 0.9, 10000.0)
        self.assertGreater(n, 100.0)
