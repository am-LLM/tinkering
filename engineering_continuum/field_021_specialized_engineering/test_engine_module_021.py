import unittest
from engine_module_021 import SpecializedEngine021

class TestEngine021(unittest.TestCase):
    def test_transfer(self):
        eng = SpecializedEngine021(1.5)
        self.assertAlmostEqual(eng.compute_transfer(0.0), 0.0)
        self.assertGreater(eng.compute_transfer(1.0), 0.5)

    def test_energy(self):
        eng = SpecializedEngine021()
        e = eng.evaluate_energy_conservation([1.0, 2.0, 3.0])
        self.assertEqual(e, 7.0)
