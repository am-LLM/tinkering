import numpy as np
from hessian_gradient_descent import HessianOptimizer

def test_newton():
    x = np.array([1.0, 1.0])
    g = np.array([2.0, 4.0])
    h = np.array([[2.0, 0.0], [0.0, 4.0]])
    x_next = HessianOptimizer.newton_step(x, g, h)
    assert np.allclose(x_next, [0.0, 0.0])
