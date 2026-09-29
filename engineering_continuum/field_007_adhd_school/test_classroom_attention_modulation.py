import unittest
from classroom_attention_modulation import AttentionModulationEngine

class TestAttention(unittest.TestCase):
    def test_salience(self):
        eng = AttentionModulationEngine(0.6)
        att = eng.step_attention(0.8, 30.0)
        self.assertGreater(att, 0.4)
