import numpy as np
from quintic_polynomial_planner import QuinticPolynomialPlanner

def test_quintic():
    coeffs = QuinticPolynomialPlanner.solve_boundary(0, 0, 0, 10, 0, 0, 5.0)
    assert len(coeffs) == 6
