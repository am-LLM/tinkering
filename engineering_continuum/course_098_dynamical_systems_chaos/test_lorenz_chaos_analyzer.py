import numpy as np
from lorenz_chaos_analyzer import LorenzChaosAnalyzer

def test_lorenz_step():
    analyzer = LorenzChaosAnalyzer()
    state = np.array([1.0, 1.0, 1.0])
    for _ in range(100):
        state = analyzer.rk4_step(state, dt=0.01)
    assert state.shape == (3,)
