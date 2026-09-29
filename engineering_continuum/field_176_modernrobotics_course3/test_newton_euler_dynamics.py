import numpy as np
from newton_euler_dynamics import NewtonEulerDynamics

def test_newton_euler():
    f, tau = NewtonEulerDynamics.link_force_torque(5.0, np.eye(3), np.array([0, 0, 9.81]), np.zeros(3), np.zeros(3))
    assert abs(f[2] - 49.05) < 1e-4
    assert np.allclose(tau, 0.0)
