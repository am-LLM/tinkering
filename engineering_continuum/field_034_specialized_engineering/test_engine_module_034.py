import unittest
from engine_module_034 import SpecializedEngine034

class TestEngine034(unittest.TestCase):
    def test_transfer(self):
        eng = SpecializedEngine034(1.5)
        self.assertAlmostEqual(eng.compute_transfer(0.0), 0.0)
        self.assertGreater(eng.compute_transfer(1.0), 0.5)

    def test_energy(self):
        eng = SpecializedEngine034()
        e = eng.evaluate_energy_conservation([1.0, 2.0, 3.0])
        self.assertEqual(e, 7.0)
