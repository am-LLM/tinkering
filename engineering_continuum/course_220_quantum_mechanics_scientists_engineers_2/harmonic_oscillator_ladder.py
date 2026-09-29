"""Course 220: Quantum Harmonic Oscillator Annihilation & Creation Ladder Operators"""
import numpy as np

class HarmonicOscillatorLadder:
    @staticmethod
    def ladder_operators(dim: int = 5) -> tuple:
        # a |n> = sqrt(n) |n-1>
        a = np.zeros((dim, dim), dtype=float)
        for n in range(1, dim):
            a[n-1, n] = np.sqrt(n)
        a_dagger = a.T
        # Hamiltonian H = hbar*omega * (a_dagger @ a + 0.5 * I)
        num_op = a_dagger @ a
        return a, a_dagger, num_op
