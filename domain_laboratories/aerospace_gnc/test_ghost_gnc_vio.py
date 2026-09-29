"""
Unit Tests for Ghost GNC: 15-State ES-EKF Visual-Inertial Odometry
"""

import pytest
import numpy as np
from unittest.mock import patch
from ghost_gnc_vio import (
    GhostESEKF, ESEKFConfig, skew_symmetric, quat_normalize,
    quat_multiply, quat_to_rot_matrix, rot_vec_to_quat, quat_slerp
)


def test_quaternion_math():
    v = np.array([1.0, 2.0, 3.0])
    vx = skew_symmetric(v)
    assert np.allclose(vx, -vx.T)
    assert np.allclose(vx @ v, np.zeros(3))
    
    # Zero norm quaternion normalization
    q_zero = quat_normalize(np.zeros(4))
    assert np.allclose(q_zero, np.array([1.0, 0.0, 0.0, 0.0]))
    
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    q_test = quat_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
    assert np.allclose(quat_multiply(q_id, q_test), q_test)
    assert np.allclose(quat_multiply(q_test, q_id), q_test)
    
    R = quat_to_rot_matrix(q_id)
    assert np.allclose(R, np.eye(3))
    
    q_sm = rot_vec_to_quat(np.array([1e-10, 0, 0]))
    assert np.isclose(q_sm[0], 1.0)
    
    rot_vec_z = np.array([0.0, 0.0, np.pi / 2.0])
    q_z = rot_vec_to_quat(rot_vec_z)
    R_z = quat_to_rot_matrix(q_z)
    vec_x = np.array([1.0, 0.0, 0.0])
    rotated_vec = R_z @ vec_x
    assert np.allclose(rotated_vec, np.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_quat_slerp():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = rot_vec_to_quat(np.array([0.0, 0.0, np.pi / 2.0]))
    
    assert np.allclose(quat_slerp(q0, q1, 0.0), q0)
    assert np.allclose(quat_slerp(q0, q1, 1.0), q1)
    
    q_mid = quat_slerp(q0, q1, 0.5)
    expected_mid = rot_vec_to_quat(np.array([0.0, 0.0, np.pi / 4.0]))
    assert np.allclose(q_mid, expected_mid)
    
    assert np.allclose(quat_slerp(q0, -q1, 1.0), q1)
    
    q_near = np.array([0.9999999, 0.0, 0.0, 0.0001])
    q_interp = quat_slerp(q0, q_near, 0.5)
    assert np.isclose(np.linalg.norm(q_interp), 1.0)


def test_esekf_imu_propagation():
    ekf = GhostESEKF()
    a_meas = np.array([0.0, 0.0, 9.80665])
    w_meas = np.array([0.0, 0.0, 0.0])
    dt = 0.01
    
    for _ in range(100):
        state = ekf.predict_imu(a_meas, w_meas, dt)
        
    assert np.allclose(state.p, np.zeros(3), atol=1e-4)
    assert np.allclose(state.v, np.zeros(3), atol=1e-4)
    assert np.all(np.diag(state.P) > 0.0)


def test_esekf_position_update_and_convergence():
    ekf = GhostESEKF()
    ekf.set_state(
        p=np.array([5.0, -3.0, 1.0]),
        v=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3),
        P=np.eye(15) * 20.0
    )
    
    z_pos = np.array([0.0, 0.0, 0.0])
    R_pos = np.eye(3) * 0.1
    
    initial_cov_trace = np.trace(ekf.state.P[0:3, 0:3])
    accepted = ekf.update_position(z_pos, R_pos)
    assert accepted is True
    assert np.linalg.norm(ekf.state.p) < 5.0
    post_cov_trace = np.trace(ekf.state.P[0:3, 0:3])
    assert post_cov_trace < initial_cov_trace

    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("Singular")):
        assert ekf.update_position(z_pos, R_pos) is False


def test_esekf_mahalanobis_anti_spoofing_gating():
    ekf = GhostESEKF()
    ekf.state.p = np.array([0.0, 0.0, 0.0])
    ekf.state.P[0:3, 0:3] = np.eye(3) * 0.1
    
    spoofed_pos = np.array([500.0, -300.0, 1000.0])
    R_pos = np.eye(3) * 0.1
    
    accepted = ekf.update_position(spoofed_pos, R_pos)
    assert accepted is False
    assert ekf.gated_measurements_count == 1
    assert np.allclose(ekf.state.p, np.zeros(3))


def test_esekf_optical_flow_velocity_update():
    ekf = GhostESEKF()
    ekf.state.v = np.array([2.0, 2.0, 0.0])
    ekf.state.P[3:6, 3:6] = np.eye(3) * 10.0
    
    z_vel = np.array([0.0, 0.0, 0.0])
    R_vel = np.eye(3) * 0.05
    
    accepted = ekf.update_velocity(z_vel, R_vel)
    assert accepted is True
    assert np.linalg.norm(ekf.state.v) < 2.0
    
    assert ekf.update_velocity(np.array([100.0, 100.0, 100.0]), R_vel) is False

    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("Singular")):
        assert ekf.update_velocity(z_vel, R_vel) is False


def test_esekf_6dof_pose_update():
    ekf = GhostESEKF()
    ekf.state.P[0:3, 0:3] = np.eye(3) * 5.0
    ekf.state.P[6:9, 6:9] = np.eye(3) * 1.0
    
    z_pos = np.array([0.2, 0.3, 0.1])
    z_q = rot_vec_to_quat(np.array([0.02, 0.0, 0.0]))
    R_pose = np.eye(6) * 0.05
    
    accepted = ekf.update_pose(z_pos, z_q, R_pose)
    assert accepted is True
    assert ekf.accepted_measurements_count > 0
    
    z_q_neg = -z_q
    accepted_neg = ekf.update_pose(z_pos, z_q_neg, R_pose)
    assert accepted_neg is True
    
    assert ekf.update_pose(np.array([1000.0, 0, 0]), z_q, R_pose) is False

    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("Singular")):
        assert ekf.update_pose(z_pos, z_q, R_pose) is False
