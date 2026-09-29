import unittest
from engine_module_038 import SpecializedEngine038

class TestEngine038(unittest.TestCase):
    def test_transfer(self):
        eng = SpecializedEngine038(1.5)
        self.assertAlmostEqual(eng.compute_transfer(0.0), 0.0)
        self.assertGreater(eng.compute_transfer(1.0), 0.5)

    def test_energy(self):
        eng = SpecializedEngine038()
        e = eng.evaluate_energy_conservation([1.0, 2.0, 3.0])
        self.assertEqual(e, 7.0)
