"""
Tri-Modal Unjammable Navigation: Fusing Magnetic Anomalies, Celestial Ephemerides, and Optical Flow
-------------------------------------------------------------------------------------------------
GPS/GNSS-Denied, EW-hardened navigation filter that unifies:
1. Magnetic Anomaly Navigation (MAGNAV): Scalar/Vector magnetic crustal anomaly map matching & gradient observation.
2. Celestial Navigation (CELNAV): Star-tracker line-of-sight unit vector ephemeris observations and TRIAD/Wahba attitude updates.
3. Optical Flow Odometry (OPTNAV): Ground-relative translational velocity and scale-corrected feature optical flow.

Unified State Vector (18D Nominal / 18D Error State):
  p: Position [x, y, z] (3D)
  v: Velocity [vx, vy, vz] (3D)
  q: Attitude quaternion [qw, qx, qy, qz] (Hamilton unit quaternion, Body -> World)
  ba: Accelerometer bias [bax, bay, baz] (3D)
  bg: Gyroscope bias [bgx, bgy, bgz] (3D)
  bm: Magnetometer anomaly bias [bmx, bmy, bmz] (3D)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List, Callable


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Computes the 3x3 skew-symmetric cross-product matrix [v]_x."""
    v = np.asarray(v, dtype=np.float64).flatten()
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalizes quaternion [qw, qx, qy, qz] to unit length."""
    q = np.asarray(q, dtype=np.float64).flatten()
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product q1 * q2, where q = [w, x, y, z]."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).flatten()
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64).flatten()
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Converts unit quaternion [w, x, y, z] to 3x3 rotation matrix R (Body -> World)."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)


def rot_vec_to_quat(rot_vec: np.ndarray) -> np.ndarray:
    """Converts a 3D rotation vector to a unit quaternion [w, x, y, z]."""
    rot_vec = np.asarray(rot_vec, dtype=np.float64).flatten()
    angle = np.linalg.norm(rot_vec)
    if angle < 1e-8:
        return np.array([1.0, 0.5 * rot_vec[0], 0.5 * rot_vec[1], 0.5 * rot_vec[2]], dtype=np.float64)
    axis = rot_vec / angle
    half_angle = 0.5 * angle
    sin_half = np.sin(half_angle)
    return np.array([np.cos(half_angle), axis[0]*sin_half, axis[1]*sin_half, axis[2]*sin_half], dtype=np.float64)


@dataclass
class TriModalNavConfig:
    sigma_acc: float = 0.04
    sigma_gyro: float = 0.004
    sigma_acc_bias: float = 0.0005
    sigma_gyro_bias: float = 0.00005
    sigma_mag_bias: float = 0.001
    
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.80665], dtype=np.float64))
    
    # Chi-Square gating thresholds (probability 0.99)
    chi2_gate_mag: float = 11.34    # 3-DOF
    chi2_gate_star: float = 9.21    # 2-DOF per star vector
    chi2_gate_opt: float = 11.34    # 3-DOF velocity


@dataclass
class StarCatalogEntry:
    star_id: str
    ra_rad: float       # Right Ascension (radians)
    dec_rad: float      # Declination (radians)
    magnitude: float    # Visual magnitude

    @property
    def inertial_unit_vector(self) -> np.ndarray:
        """Returns inertial unit line-of-sight vector in J2000 / ECI frame."""
        return np.array([
            np.cos(self.dec_rad) * np.cos(self.ra_rad),
            np.cos(self.dec_rad) * np.sin(self.ra_rad),
            np.sin(self.dec_rad)
        ], dtype=np.float64)


class CrustalMagneticMap:
    """
    Synthetic / Interpolated Earth crustal magnetic anomaly field model.
    Provides vector anomaly field B_world(p) [nT] and spatial gradient tensor dB/dp.
    """
    def __init__(self, base_field: Optional[np.ndarray] = None, anomaly_amplitude: float = 250.0):
        self.base_field = base_field if base_field is not None else np.array([22000.0, -1500.0, 42000.0], dtype=np.float64)
        self.amplitude = float(anomaly_amplitude)
        self.spatial_freq = 0.001  # Anomaly spatial variation frequency (1/m)

    def evaluate_field_and_gradient(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates B_world(p) and 3x3 spatial Jacobian H_p = d(B_world)/d(p).
        """
        p = np.asarray(p, dtype=np.float64).flatten()
        k = self.spatial_freq
        
        # Non-linear synthetic crustal anomaly
        s_x = np.sin(k * p[0])
        c_y = np.cos(k * p[1])
        s_z = np.sin(k * p[2] * 0.5)
        
        b_anom = self.amplitude * np.array([
            s_x * c_y,
            np.cos(k * p[0]) * np.sin(k * p[1]),
            s_x * s_z
        ], dtype=np.float64)
        
        B_total = self.base_field + b_anom
        
        # Spatial Gradient Tensor dB/dp
        dB_dp = self.amplitude * k * np.array([
            [np.cos(k*p[0]) * c_y, -s_x * np.sin(k*p[1]), 0.0],
            [-np.sin(k*p[0]) * np.sin(k*p[1]), np.cos(k*p[0]) * np.cos(k*p[1]), 0.0],
            [np.cos(k*p[0]) * s_z, 0.0, 0.5 * s_x * np.cos(k*p[2]*0.5)]
        ], dtype=np.float64)
        
        return B_total, dB_dp


class TriModalNavFilter:
    """
    18-State Error-State Kalman Filter fusing IMU, Magnetic Anomaly Matching,
    Celestial Star Trackers, and Optical Flow Visual Odometry.
    """
    def __init__(self, config: Optional[TriModalNavConfig] = None, mag_map: Optional[CrustalMagneticMap] = None):
        self.config = config or TriModalNavConfig()
        self.mag_map = mag_map or CrustalMagneticMap()
        
        # State vector components
        self.p = np.zeros(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.ba = np.zeros(3, dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)
        self.bm = np.zeros(3, dtype=np.float64)
        
        # 18x18 Covariance Matrix P
        self.P = np.diag([
            1.0, 1.0, 1.0,         # pos (m^2)
            0.1, 0.1, 0.1,         # vel (m^2/s^2)
            0.01, 0.01, 0.01,      # rot (rad^2)
            0.01, 0.01, 0.01,      # ba (m^2/s^4)
            0.001, 0.001, 0.001,   # bg (rad^2/s^2)
            10.0, 10.0, 10.0       # bm (nT^2)
        ]).astype(np.float64)
        
        # Telemetry
        self.accepted_mag_count = 0
        self.rejected_mag_count = 0
        self.accepted_star_count = 0
        self.rejected_star_count = 0
        self.accepted_opt_count = 0
        self.rejected_opt_count = 0

    def set_state(self, p: np.ndarray, v: np.ndarray, q: np.ndarray,
                  ba: Optional[np.ndarray] = None, bg: Optional[np.ndarray] = None, bm: Optional[np.ndarray] = None,
                  P: Optional[np.ndarray] = None):
        self.p = np.asarray(p, dtype=np.float64).flatten()
        self.v = np.asarray(v, dtype=np.float64).flatten()
        self.q = quat_normalize(np.asarray(q, dtype=np.float64).flatten())
        if ba is not None: self.ba = np.asarray(ba, dtype=np.float64).flatten()
        if bg is not None: self.bg = np.asarray(bg, dtype=np.float64).flatten()
        if bm is not None: self.bm = np.asarray(bm, dtype=np.float64).flatten()
        if P is not None: self.P = np.asarray(P, dtype=np.float64).copy()

    def predict_imu(self, acc_meas: np.ndarray, gyro_meas: np.ndarray, dt: float):
        """High-rate strapdown IMU dead reckoning + 18-state error covariance propagation."""
        assert dt > 0, "dt must be > 0"
        acc_meas = np.asarray(acc_meas, dtype=np.float64).flatten()
        gyro_meas = np.asarray(gyro_meas, dtype=np.float64).flatten()
        
        acc_u = acc_meas - self.ba
        gyro_u = gyro_meas - self.bg
        
        R = quat_to_rot_matrix(self.q)
        acc_w = R @ acc_u + self.config.gravity
        
        # State integration
        self.p += self.v * dt + 0.5 * acc_w * (dt ** 2)
        self.v += acc_w * dt
        
        dq = rot_vec_to_quat(gyro_u * dt)
        self.q = quat_normalize(quat_multiply(self.q, dq))
        
        # 18x18 Error State Transition Matrix F_x
        # States: [delta_p(0:3), delta_v(3:6), delta_theta(6:9), delta_ba(9:12), delta_bg(12:15), delta_bm(15:18)]
        F_x = np.eye(18, dtype=np.float64)
        F_x[0:3, 3:6] = np.eye(3) * dt
        F_x[3:6, 6:9] = -R @ skew_symmetric(acc_u) * dt
        F_x[3:6, 9:12] = -R * dt
        
        omega_skew = skew_symmetric(gyro_u)
        F_x[6:9, 6:9] = np.eye(3) - omega_skew * dt
        F_x[6:9, 12:15] = -np.eye(3) * dt
        
        # Process Noise Covariance
        Q_d = np.zeros((18, 18), dtype=np.float64)
        Q_d[3:6, 3:6] = np.eye(3) * ((self.config.sigma_acc * dt) ** 2)
        Q_d[6:9, 6:9] = np.eye(3) * ((self.config.sigma_gyro * dt) ** 2)
        Q_d[9:12, 9:12] = np.eye(3) * ((self.config.sigma_acc_bias * dt) ** 2)
        Q_d[12:15, 12:15] = np.eye(3) * ((self.config.sigma_gyro_bias * dt) ** 2)
        Q_d[15:18, 15:18] = np.eye(3) * ((self.config.sigma_mag_bias * dt) ** 2)
        
        self.P = F_x @ self.P @ F_x.T + Q_d
        self.P = 0.5 * (self.P + self.P.T)

    def update_magnetic_anomaly(self, z_mag_body: np.ndarray, R_mag: np.ndarray) -> bool:
        """
        MAGNAV update: Compares measured 3-axis magnetometer reading in body frame against
        projected crustal magnetic map vector B_world(p).
        Measurement Model: z_mag = R^T * B_world(p) + b_m + v_mag
        """
        z_mag_body = np.asarray(z_mag_body, dtype=np.float64).flatten()
        R_mag = np.asarray(R_mag, dtype=np.float64)
        
        R_b2w = quat_to_rot_matrix(self.q)
        R_w2b = R_b2w.T
        
        B_w_est, dB_dp = self.mag_map.evaluate_field_and_gradient(self.p)
        z_mag_est = R_w2b @ B_w_est + self.bm
        
        residual = z_mag_body - z_mag_est
        
        # Measurement Jacobian H (3x18)
        # d(z_mag)/d(delta_p) = R_w2b * (dB_dp)
        # d(z_mag)/d(delta_theta) = [R_w2b * B_w_est]_x
        # d(z_mag)/d(delta_bm) = I_3
        H = np.zeros((3, 18), dtype=np.float64)
        H[0:3, 0:3] = R_w2b @ dB_dp
        H[0:3, 6:9] = skew_symmetric(R_w2b @ B_w_est)
        H[0:3, 15:18] = np.eye(3)
        
        S = H @ self.P @ H.T + R_mag
        try:
            S_inv = np.linalg.inv(S)
            d_m2 = float(residual.T @ S_inv @ residual)
        except np.linalg.LinAlgError:
            self.rejected_mag_count += 1
            return False
            
        if d_m2 > self.config.chi2_gate_mag:
            self.rejected_mag_count += 1
            return False
            
        K = self.P @ H.T @ S_inv
        delta_x = K @ residual
        self._inject_error_state(delta_x, K, H, R_mag)
        self.accepted_mag_count += 1
        return True

    def update_celestial_star(self, measured_unit_body: np.ndarray, catalog_entry: StarCatalogEntry, R_star: np.ndarray) -> bool:
        """
        CELNAV update: Compares measured unit vector to a detected star in body frame
        against the catalog star inertial unit vector.
        Measurement Model: s_body = R^T * s_inertial + v_star
        """
        s_body_meas = np.asarray(measured_unit_body, dtype=np.float64).flatten()
        norm_s = np.linalg.norm(s_body_meas)
        if norm_s < 1e-6:
            return False
        s_body_meas = s_body_meas / norm_s
        
        R_star = np.asarray(R_star, dtype=np.float64)
        s_inertial = catalog_entry.inertial_unit_vector
        
        R_b2w = quat_to_rot_matrix(self.q)
        R_w2b = R_b2w.T
        
        s_body_pred = R_w2b @ s_inertial
        residual = s_body_meas - s_body_pred
        
        # Jacobian H (3x18)
        # d(s_body)/d(delta_theta) = [s_body_pred]_x
        H = np.zeros((3, 18), dtype=np.float64)
        H[0:3, 6:9] = skew_symmetric(s_body_pred)
        
        S = H @ self.P @ H.T + R_star
        try:
            S_inv = np.linalg.inv(S)
            d_m2 = float(residual.T @ S_inv @ residual)
        except np.linalg.LinAlgError:
            self.rejected_star_count += 1
            return False
            
        if d_m2 > self.config.chi2_gate_star:
            self.rejected_star_count += 1
            return False
            
        K = self.P @ H.T @ S_inv
        delta_x = K @ residual
        self._inject_error_state(delta_x, K, H, R_star)
        self.accepted_star_count += 1
        return True

    def update_optical_flow_velocity(self, z_vel_world: np.ndarray, R_opt: np.ndarray) -> bool:
        """
        OPTNAV update: Fuses ground-relative translational velocity estimated from optical flow.
        Measurement Model: z_opt = v + v_opt
        """
        z_vel_world = np.asarray(z_vel_world, dtype=np.float64).flatten()
        R_opt = np.asarray(R_opt, dtype=np.float64)
        
        residual = z_vel_world - self.v
        
        H = np.zeros((3, 18), dtype=np.float64)
        H[0:3, 3:6] = np.eye(3)
        
        S = H @ self.P @ H.T + R_opt
        try:
            S_inv = np.linalg.inv(S)
            d_m2 = float(residual.T @ S_inv @ residual)
        except np.linalg.LinAlgError:
            self.rejected_opt_count += 1
            return False
            
        if d_m2 > self.config.chi2_gate_opt:
            self.rejected_opt_count += 1
            return False
            
        K = self.P @ H.T @ S_inv
        delta_x = K @ residual
        self._inject_error_state(delta_x, K, H, R_opt)
        self.accepted_opt_count += 1
        return True

    def _inject_error_state(self, delta_x: np.ndarray, K: np.ndarray, H: np.ndarray, R_meas: np.ndarray):
        """Injects error state into nominal states and executes Joseph-form covariance reset."""
        dp = delta_x[0:3]
        dv = delta_x[3:6]
        dtheta = delta_x[6:9]
        dba = delta_x[9:12]
        dbg = delta_x[12:15]
        dbm = delta_x[15:18]
        
        self.p += dp
        self.v += dv
        self.ba += dba
        self.bg += dbg
        self.bm += dbm
        
        dq = rot_vec_to_quat(dtheta)
        self.q = quat_normalize(quat_multiply(self.q, dq))
        
        # Joseph Form Covariance Update
        I_KH = np.eye(18, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T
        
        # Reset matrix G for orientation
        G = np.eye(18, dtype=np.float64)
        G[6:9, 6:9] = np.eye(3) - 0.5 * skew_symmetric(dtheta)
        self.P = G @ self.P @ G.T
        self.P = 0.5 * (self.P + self.P.T)


def triad_attitude_determination(v1_inertial: np.ndarray, v2_inertial: np.ndarray,
                                 w1_body: np.ndarray, w2_body: np.ndarray) -> np.ndarray:
    """
    Deterministic TRIAD algorithm for complete 3-axis attitude determination
    from two non-parallel line-of-sight unit vector observations.
    Returns 3x3 rotation matrix R (Body -> Inertial).
    """
    v1 = v1_inertial / np.linalg.norm(v1_inertial)
    v2 = v2_inertial / np.linalg.norm(v2_inertial)
    w1 = w1_body / np.linalg.norm(w1_body)
    w2 = w2_body / np.linalg.norm(w2_body)
    
    # Inertial frame orthonormal triad
    r1 = v1
    r2 = np.cross(v1, v2)
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    M_inertial = np.column_stack([r1, r2, r3])
    
    # Body frame orthonormal triad
    s1 = w1
    s2 = np.cross(w1, w2)
    s2 = s2 / np.linalg.norm(s2)
    s3 = np.cross(s1, s2)
    M_body = np.column_stack([s1, s2, s3])
    
    # R_b2i = M_inertial @ M_body.T
    R_b2i = M_inertial @ M_body.T
    return R_b2i
