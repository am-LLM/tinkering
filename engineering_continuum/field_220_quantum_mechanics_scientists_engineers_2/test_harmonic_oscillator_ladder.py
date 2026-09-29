import numpy as np
from harmonic_oscillator_ladder import HarmonicOscillatorLadder

def test_ladder():
    a, a_dag, num_op = HarmonicOscillatorLadder.ladder_operators(5)
    comm = a @ a_dag - a_dag @ a
    # [a, a^dagger] = I
    assert np.allclose(comm[:4, :4], np.eye(4))
