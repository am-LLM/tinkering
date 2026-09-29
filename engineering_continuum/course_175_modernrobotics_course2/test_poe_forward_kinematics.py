import numpy as np
from poe_forward_kinematics import PoEForwardKinematics

def test_poe():
    s = np.array([0, 0, 1, 0, 0, 0])
    t = PoEForwardKinematics.matrix_exp6(s, np.pi/2)
    assert abs(t[0, 1] - (-1.0)) < 1e-4
