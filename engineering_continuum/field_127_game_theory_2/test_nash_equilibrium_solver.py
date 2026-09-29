import numpy as np
from nash_equilibrium_solver import NashEquilibriumSolver

def test_matching_pennies():
    a = np.array([[1, -1], [-1, 1]])
    b = -a
    p, q = NashEquilibriumSolver.solve_2x2_mixed(a, b)
    assert abs(p - 0.5) < 1e-4
    assert abs(q - 0.5) < 1e-4
