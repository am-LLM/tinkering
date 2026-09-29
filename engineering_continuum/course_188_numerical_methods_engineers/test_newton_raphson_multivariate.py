import numpy as np
from newton_raphson_multivariate import MultivariateNewtonRaphson

def test_multivariate_newton():
    # f(x, y) = [x^2 + y^2 - 4, x - y]
    f = lambda v: np.array([v[0]**2 + v[1]**2 - 4.0, v[0] - v[1]])
    j = lambda v: np.array([[2.0*v[0], 2.0*v[1]], [1.0, -1.0]])
    sol = MultivariateNewtonRaphson.solve(f, j, np.array([1.0, 1.0]))
    assert np.allclose(sol, [np.sqrt(2), np.sqrt(2)])
