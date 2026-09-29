import numpy as np
from kelly_sharpe_optimizer import KellySharpeOptimizer

def test_kelly():
    f = KellySharpeOptimizer.kelly_fraction(0.6, 2.0)
    assert abs(f - 0.4) < 1e-4
