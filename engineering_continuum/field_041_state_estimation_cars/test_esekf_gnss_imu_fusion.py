import numpy as np
from esekf_gnss_imu_fusion import ESEKFFusion

def test_esekf_gnss_fusion():
    filter_node = ESEKFFusion(dt=0.1)
    for _ in range(10):
        filter_node.predict(accel=np.array([1.0, 0.0, 0.0]))
    pos = filter_node.update_gnss(gnss_pos=np.array([0.5, 0.0, 0.0]))
    assert pos[0] > 0.0
    assert np.all(np.diag(filter_node.cov) > 0)
