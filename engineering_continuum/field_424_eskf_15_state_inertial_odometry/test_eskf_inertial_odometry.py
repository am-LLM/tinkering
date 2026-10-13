import pytest
import numpy as np
from engineering_continuum.field_424_eskf_15_state_inertial_odometry.eskf_inertial_odometry import ESKF15StateVisualInertialOdometry

def test_field_424_initialization():
    eskf = ESKF15StateVisualInertialOdometry(dt=0.01)
    assert eskf.P.shape == (15, 15)
    assert np.allclose(eskf.pos, np.zeros(3))
    assert eskf.quat[0] == 1.0

def test_field_424_optical_flow_fusion_zero_drift():
    eskf = ESKF15StateVisualInertialOdometry(dt=0.01)
    
    # 50 IMU prediction steps with forward velocity
    for _ in range(50):
        # 1g vertical acceleration against gravity
        eskf.predict_imu_step(np.array([0.5, 0.0, 9.81]), np.zeros(3))
        # Downward optical flow measures forward speed 0.25 m/s
        eskf.update_optical_flow_velocity(np.array([0.25, 0.0, 0.0]))
        
    assert eskf.vel[0] > 0.15 # Accurately tracks forward motion
    assert abs(eskf.vel[1]) < 0.05 # Zero lateral drift
    assert bool(eskf.verify_properties()) is True
