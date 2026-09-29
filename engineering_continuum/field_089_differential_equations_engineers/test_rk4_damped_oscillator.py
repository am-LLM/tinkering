import numpy as np
from rk4_damped_oscillator import DampedOscillatorRK4

def test_rk4():
    s = DampedOscillatorRK4()
    st = s.step(np.array([1.0, 0.0]), 0, 0.1)
    assert len(st) == 2
