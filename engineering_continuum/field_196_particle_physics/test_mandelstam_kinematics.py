import numpy as np
from mandelstam_kinematics import MandelstamKinematics

def test_mandelstam():
    p1 = np.array([5.0, 0.0, 0.0, 3.0])
    p2 = np.array([5.0, 0.0, 0.0, -3.0])
    s = MandelstamKinematics.mandelstam_s(p1, p2)
    assert abs(s - 100.0) < 1e-4
    assert MandelstamKinematics.center_of_mass_energy(s) == 10.0
