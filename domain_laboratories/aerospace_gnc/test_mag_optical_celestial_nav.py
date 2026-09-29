"""
Unit Tests for Tri-Modal Unjammable Navigation (MAGNAV + CELNAV + OPTNAV)
"""

import pytest
import numpy as np
from unittest.mock import patch
from mag_optical_celestial_nav import (
    TriModalNavFilter, TriModalNavConfig, CrustalMagneticMap,
    StarCatalogEntry, triad_attitude_determination,
    skew_symmetric, quat_normalize, quat_multiply,
    quat_to_rot_matrix, rot_vec_to_quat
)


def test_quaternion_and_skew_helpers():
    v = np.array([0.0, 0.0, 0.0])
    assert np.allclose(skew_symmetric(v), np.zeros((3, 3)))
    
    q_zero = np.zeros(4)
    q_norm = quat_normalize(q_zero)
    assert np.allclose(q_norm, np.array([1.0, 0.0, 0.0, 0.0]))

    v_small = np.array([1e-10, 0.0, 0.0])
    q_small = rot_vec_to_quat(v_small)
    assert np.isclose(q_small[0], 1.0)
    
    v_rot = np.array([0.0, np.pi/2, 0.0])
    q_rot = rot_vec_to_quat(v_rot)
    R_rot = quat_to_rot_matrix(q_rot)
    test_vec = np.array([1.0, 0.0, 0.0])
    assert np.allclose(R_rot @ test_vec, np.array([0.0, 0.0, -1.0]), atol=1e-6)


def test_crustal_magnetic_map():
    mag_map = CrustalMagneticMap(base_field=np.array([20000.0, 0.0, 40000.0]), anomaly_amplitude=300.0)
    p = np.array([100.0, 200.0, 50.0])
    B, dB_dp = mag_map.evaluate_field_and_gradient(p)
    
    assert B.shape == (3,)
    assert dB_dp.shape == (3, 3)
    assert not np.isnan(B).any()
    assert not np.isnan(dB_dp).any()


def test_star_catalog_entry():
    star = StarCatalogEntry(star_id="Polaris", ra_rad=0.0, dec_rad=np.pi/2, magnitude=2.0)
    u_vec = star.inertial_unit_vector
    assert np.allclose(u_vec, np.array([0.0, 0.0, 1.0]), atol=1e-6)


def test_triad_attitude_determination():
    v1_i = np.array([1.0, 0.0, 0.0])
    v2_i = np.array([0.0, 1.0, 0.0])
    
    R_true = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    w1_b = R_true.T @ v1_i
    w2_b = R_true.T @ v2_i
    
    R_est = triad_attitude_determination(v1_i, v2_i, w1_b, w2_b)
    assert np.allclose(R_est, R_true, atol=1e-6)
    assert np.isclose(np.linalg.det(R_est), 1.0)


def test_trimodal_nav_imu_propagation():
    filter_nav = TriModalNavFilter()
    acc_meas = np.array([0.0, 0.0, 9.80665])
    gyro_meas = np.array([0.0, 0.0, 0.0])
    dt = 0.02
    
    for _ in range(50):
        filter_nav.predict_imu(acc_meas, gyro_meas, dt)
        
    assert np.allclose(filter_nav.p, np.zeros(3), atol=1e-4)
    assert np.allclose(filter_nav.v, np.zeros(3), atol=1e-4)
    assert np.all(np.diag(filter_nav.P) > 0.0)


def test_trimodal_nav_mag_update():
    filter_nav = TriModalNavFilter()
    filter_nav.set_state(
        p=np.array([10.0, 20.0, 0.0]),
        v=np.array([0.0, 0.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        ba=np.zeros(3),
        bg=np.zeros(3),
        bm=np.zeros(3),
        P=np.eye(18) * 0.5
    )
    
    B_true, _ = filter_nav.mag_map.evaluate_field_and_gradient(filter_nav.p)
    R_mag = np.eye(3) * 5.0
    
    accepted = filter_nav.update_magnetic_anomaly(B_true + np.random.normal(0, 0.1, 3), R_mag)
    assert accepted is True
    assert filter_nav.accepted_mag_count == 1
    
    spoofed_mag = B_true + np.array([5000.0, -5000.0, 5000.0])
    rejected = filter_nav.update_magnetic_anomaly(spoofed_mag, R_mag)
    assert rejected is False
    assert filter_nav.rejected_mag_count == 1

    # Test LinAlgError exception handling branch
    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("Singular matrix")):
        assert filter_nav.update_magnetic_anomaly(B_true, R_mag) is False


def test_trimodal_nav_celestial_update():
    filter_nav = TriModalNavFilter()
    star = StarCatalogEntry(star_id="Vega", ra_rad=0.5, dec_rad=0.7, magnitude=0.03)
    s_inertial = star.inertial_unit_vector
    
    R_star = np.eye(3) * 1e-4
    
    accepted = filter_nav.update_celestial_star(s_inertial, star, R_star)
    assert accepted is True
    assert filter_nav.accepted_star_count == 1
    
    assert filter_nav.update_celestial_star(np.zeros(3), star, R_star) is False
    
    outlier_vec = np.array([-s_inertial[0], -s_inertial[1], -s_inertial[2]])
    rejected = filter_nav.update_celestial_star(outlier_vec, star, R_star)
    assert rejected is False
    assert filter_nav.rejected_star_count == 1

    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("Singular matrix")):
        assert filter_nav.update_celestial_star(s_inertial, star, R_star) is False


def test_trimodal_nav_optical_flow_update():
    filter_nav = TriModalNavFilter()
    filter_nav.v = np.array([1.0, 0.5, -0.2])
    
    R_opt = np.eye(3) * 0.01
    accepted = filter_nav.update_optical_flow_velocity(np.array([1.02, 0.49, -0.21]), R_opt)
    assert accepted is True
    assert filter_nav.accepted_opt_count == 1
    
    outlier_v = np.array([50.0, -50.0, 50.0])
    rejected = filter_nav.update_optical_flow_velocity(outlier_v, R_opt)
    assert rejected is False
    assert filter_nav.rejected_opt_count == 1

    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("Singular matrix")):
        assert filter_nav.update_optical_flow_velocity(filter_nav.v, R_opt) is False
