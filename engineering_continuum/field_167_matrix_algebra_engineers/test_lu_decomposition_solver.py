import numpy as np
from lu_decomposition_solver import LUSolver

def test_lu():
    a = np.array([[2.0, 1.0], [4.0, 5.0]])
    l, u = LUSolver.lu_decompose(a)
    assert np.allclose(l @ u, a)
