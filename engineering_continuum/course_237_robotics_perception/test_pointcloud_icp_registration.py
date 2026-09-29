import numpy as np
from pointcloud_icp_registration import PointCloudICP

def test_icp_svd():
    p = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    t_true = np.array([2.0, 3.0])
    p_trans = p + t_true
    r, t = PointCloudICP.align_svd(p, p_trans)
    assert np.allclose(r, np.eye(2))
    assert np.allclose(t, t_true)
