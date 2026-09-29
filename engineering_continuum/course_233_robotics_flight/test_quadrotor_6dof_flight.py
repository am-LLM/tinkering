import numpy as np
from quadrotor_6dof_flight import Quadrotor6DOFFlight

def test_quadrotor_hover():
    quad = Quadrotor6DOFFlight(mass=1.0)
    hover_thrust = 1.0 * 9.81
    pos = np.array([0.0, 0.0, 10.0])
    vel = np.array([0.0, 0.0, 0.0])
    pos_next, vel_next = quad.step_linear_dynamics(pos, vel, hover_thrust, 0.0, 0.0, 0.1)
    assert np.allclose(pos_next, pos)
