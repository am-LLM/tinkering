import numpy as np
from spatial_transform_se3 import SE3Transform

def test_se3():
    t = SE3Transform.build_t(np.eye(3), np.array([1, 2, 3]))
    t_inv = SE3Transform.inverse_t(t)
    assert np.allclose(t @ t_inv, np.eye(4))
