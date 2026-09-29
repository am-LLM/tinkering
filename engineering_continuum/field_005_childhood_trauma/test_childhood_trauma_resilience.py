import unittest
from childhood_trauma_resilience_model import TraumaResilienceModel

class TestTraumaResilience(unittest.TestCase):
    def test_allostatic_load(self):
        m = TraumaResilienceModel(ace_score=4)
        load = m.evaluate_allostatic_load(12.0, 1.5)
        self.assertGreater(load, 12.0)
        
    def test_resilience_index(self):
        m = TraumaResilienceModel(ace_score=2)
        idx = m.compute_resilience_index(3, 50.0)
        self.assertGreater(idx, 0.5)
