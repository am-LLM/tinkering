import numpy as np
from ransac_essential_matrix_vo import VisualOdometryRANSAC

def test_vo_pose_estimation():
    vo = VisualOdometryRANSAC()
    # Synthetic correspondence points
    p1 = np.array([[100, 150], [200, 250], [300, 180], [120, 310], [400, 220], [250, 110], [350, 400], [180, 290]])
    p2 = p1 + np.array([2.0, 1.0])
    r, t = vo.estimate_pose(p1, p2)
    assert r.shape == (3, 3)
    assert t.shape == (3,)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-2)
