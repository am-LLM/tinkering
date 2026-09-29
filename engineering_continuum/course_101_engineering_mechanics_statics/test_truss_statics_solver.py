import numpy as np
from truss_statics_solver import TrussStaticsSolver

def test_truss_joint():
    forces = TrussStaticsSolver.solve_joint_2d(0.0, -100.0, [np.pi/4, 3*np.pi/4])
    assert abs(forces[0] - forces[1]) < 1e-4
