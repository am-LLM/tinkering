"""
Ghost GNC: 15-State Error-State Extended Kalman Filter (ES-EKF) for Visual-Inertial Odometry (VIO)
--------------------------------------------------------------------------------------------------
Fuses high-rate 6-DoF IMU dead reckoning (strapdown integration) with monocular optical flow
visual odometry updates, featuring chi-square Mahalanobis distance outlier gating for anti-spoofing.

State Definition:
  Nominal State (16D):
    p: Position in World frame [x, y, z] (3D)
    v: Velocity in World frame [vx, vy, vz] (3D)
    q: Orientation Quaternion [qw, qx, qy, qz] (Hamilton unit quaternion, World -> Body or Body -> World)
    ba: Accelerometer Bias [bax, bay, baz] (3D)
    bg: Gyroscope Bias [bgx, bgy, bgz] (3D)
    
  Error State (15D):
    delta_x = [delta_p (3), delta_v (3), delta_theta (3), delta_ba (3), delta_bg (3)]^T
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List


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


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical Linear Interpolation (SLERP) between two quaternions q0 and q1.
    t in [0.0, 1.0].
    """
    q0 = quat_normalize(np.asarray(q0, dtype=np.float64))
    q1 = quat_normalize(np.asarray(q1, dtype=np.float64))
    
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
        
    dot = np.clip(dot, -1.0, 1.0)
    
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return quat_normalize(result)
        
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    
    return quat_normalize(s0 * q0 + s1 * q1)


def rot_vec_to_quat(rot_vec: np.ndarray) -> np.ndarray:
    """Converts a 3D rotation vector (axis * angle) to a unit quaternion [w, x, y, z]."""
    rot_vec = np.asarray(rot_vec, dtype=np.float64).flatten()
    angle = np.linalg.norm(rot_vec)
    if angle < 1e-8:
        return np.array([1.0, 0.5 * rot_vec[0], 0.5 * rot_vec[1], 0.5 * rot_vec[2]], dtype=np.float64)
    axis = rot_vec / angle
    half_angle = 0.5 * angle
    sin_half = np.sin(half_angle)
    return np.array([np.cos(half_angle), axis[0]*sin_half, axis[1]*sin_half, axis[2]*sin_half], dtype=np.float64)


@dataclass
class ESEKFConfig:
    # Process Noise Spectral Densities
    sigma_acc_noise: float = 0.05       # [m/s^2 / sqrt(Hz)] Accelerometer noise
    sigma_gyro_noise: float = 0.005     # [rad/s / sqrt(Hz)] Gyroscope noise
    sigma_acc_bias_walk: float = 0.001  # [m/s^3 / sqrt(Hz)] Accelerometer bias random walk
    sigma_gyro_bias_walk: float = 0.0001# [rad/s^2 / sqrt(Hz)] Gyro bias random walk
    
    # Gravity vector in world frame (Z-up: [0, 0, -9.80665])
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.80665], dtype=np.float64))
    
    # Chi-Square gating thresholds
    chi2_gate_pos: float = 12.0
    chi2_gate_vel: float = 12.0
    chi2_gate_pose: float = 20.0


@dataclass
class FilterState:
    p: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    P: np.ndarray = field(default_factory=lambda: np.eye(15, dtype=np.float64) * 0.1)


class GhostESEKF:
    """
    15-State Error-State Kalman Filter for GPS-Denied Visual-Inertial Navigation.
    """
    def __init__(self, config: Optional[ESEKFConfig] = None):
        self.config = config or ESEKFConfig()
        self.state = FilterState()
        
        # Initialize default covariance matrix P
        self.state.P = np.diag([
            0.01, 0.01, 0.01,   # pos (m^2)
            0.01, 0.01, 0.01,   # vel (m^2/s^2)
            0.001, 0.001, 0.001,# rot (rad^2)
            0.005, 0.005, 0.005,# ba (m^2/s^4)
            0.0005, 0.0005, 0.0005 # bg (rad^2/s^2)
        ]).astype(np.float64)
        
        # Performance & Telemetry counters
        self.step_count: int = 0
        self.gated_measurements_count: int = 0
        self.accepted_measurements_count: int = 0
        self.last_mahalanobis_dist: float = 0.0

    def set_state(self, p: np.ndarray, v: np.ndarray, q: np.ndarray, ba: np.ndarray, bg: np.ndarray, P: Optional[np.ndarray] = None):
        """Manually sets filter nominal states and covariance."""
        self.state.p = np.array(p, dtype=np.float64).flatten()
        self.state.v = np.array(v, dtype=np.float64).flatten()
        self.state.q = quat_normalize(np.array(q, dtype=np.float64).flatten())
        self.state.ba = np.array(ba, dtype=np.float64).flatten()
        self.state.bg = np.array(bg, dtype=np.float64).flatten()
        if P is not None:
            self.state.P = np.array(P, dtype=np.float64).copy()

    def predict_imu(self, acc_meas: np.ndarray, gyro_meas: np.ndarray, dt: float) -> FilterState:
        """
        Strapdown IMU dead reckoning state propagation + 15-state error covariance propagation.
        """
        assert dt > 0, "dt must be positive"
        acc_meas = np.asarray(acc_meas, dtype=np.float64).flatten()
        gyro_meas = np.asarray(gyro_meas, dtype=np.float64).flatten()
        
        # 1. Correct IMU measurements with current bias estimates
        acc_unbiased = acc_meas - self.state.ba
        gyro_unbiased = gyro_meas - self.state.bg
        
        # Current rotation matrix R_b_to_w
        R = quat_to_rot_matrix(self.state.q)
        
        # Specific acceleration in world frame
        acc_world = R @ acc_unbiased + self.config.gravity
        
        # 2. Kinematic state propagation (Midpoint / Euler integration)
        self.state.p = self.state.p + self.state.v * dt + 0.5 * acc_world * (dt ** 2)
        self.state.v = self.state.v + acc_world * dt
        
        # Orientation integration via delta quaternion
        delta_rot = gyro_unbiased * dt
        dq = rot_vec_to_quat(delta_rot)
        self.state.q = quat_normalize(quat_multiply(self.state.q, dq))
        
        # 3. Continuous-to-discrete Error State Transition Matrix F_x (15x15)
        F_x = np.eye(15, dtype=np.float64)
        F_x[0:3, 3:6] = np.eye(3) * dt
        F_x[3:6, 6:9] = -R @ skew_symmetric(acc_unbiased) * dt
        F_x[3:6, 9:12] = -R * dt
        
        omega_skew = skew_symmetric(gyro_unbiased)
        F_x[6:9, 6:9] = np.eye(3) - omega_skew * dt
        F_x[6:9, 12:15] = -np.eye(3) * dt
        
        # 4. Continuous-to-Discrete Process Noise Covariance Q_d
        Q_c = np.zeros((12, 12), dtype=np.float64)
        Q_c[0:3, 0:3] = np.eye(3) * (self.config.sigma_acc_noise ** 2)
        Q_c[3:6, 3:6] = np.eye(3) * (self.config.sigma_gyro_noise ** 2)
        Q_c[6:9, 6:9] = np.eye(3) * (self.config.sigma_acc_bias_walk ** 2)
        Q_c[9:12, 9:12] = np.eye(3) * (self.config.sigma_gyro_bias_walk ** 2)
        
        F_i = np.zeros((15, 12), dtype=np.float64)
        F_i[3:6, 0:3] = -R
        F_i[6:9, 3:6] = -np.eye(3)
        F_i[9:12, 6:9] = np.eye(3)
        F_i[12:15, 9:12] = np.eye(3)
        
        Q_d = (F_i @ Q_c @ F_i.T) * dt
        
        # 5. Covariance propagation
        self.state.P = F_x @ self.state.P @ F_x.T + Q_d
        self.state.P = 0.5 * (self.state.P + self.state.P.T)
        
        self.step_count += 1
        return self.state

    def update_position(self, z_pos: np.ndarray, R_pos: np.ndarray) -> bool:
        """
        Measurement update for direct 3D position (e.g. Visual Odometry translation).
        Includes Mahalanobis distance gating for anti-spoofing rejection.
        """
        z_pos = np.asarray(z_pos, dtype=np.float64).flatten()
        R_pos = np.asarray(R_pos, dtype=np.float64)
        
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        
        residual = z_pos - self.state.p
        S = H @ self.state.P @ H.T + R_pos
        
        try:
            S_inv = np.linalg.inv(S)
            d_mahalanobis_sq = float(residual.T @ S_inv @ residual)
        except np.linalg.LinAlgError:
            return False
            
        self.last_mahalanobis_dist = float(np.sqrt(max(0.0, d_mahalanobis_sq)))
        
        if d_mahalanobis_sq > self.config.chi2_gate_pos:
            self.gated_measurements_count += 1
            return False
            
        K = self.state.P @ H.T @ S_inv
        delta_x = K @ residual
        self._inject_error_and_reset(delta_x, K, H, R_pos)
        self.accepted_measurements_count += 1
        return True

    def update_velocity(self, z_vel: np.ndarray, R_vel: np.ndarray) -> bool:
        """
        Measurement update for 3D velocity (e.g. monocular optical flow velocity estimate).
        Includes Mahalanobis gating.
        """
        z_vel = np.asarray(z_vel, dtype=np.float64).flatten()
        R_vel = np.asarray(R_vel, dtype=np.float64)
        
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 3:6] = np.eye(3)
        
        residual = z_vel - self.state.v
        S = H @ self.state.P @ H.T + R_vel
        
        try:
            S_inv = np.linalg.inv(S)
            d_mahalanobis_sq = float(residual.T @ S_inv @ residual)
        except np.linalg.LinAlgError:
            return False
            
        self.last_mahalanobis_dist = float(np.sqrt(max(0.0, d_mahalanobis_sq)))
        
        if d_mahalanobis_sq > self.config.chi2_gate_vel:
            self.gated_measurements_count += 1
            return False
            
        K = self.state.P @ H.T @ S_inv
        delta_x = K @ residual
        self._inject_error_and_reset(delta_x, K, H, R_vel)
        self.accepted_measurements_count += 1
        return True

    def update_pose(self, z_pos: np.ndarray, z_q: np.ndarray, R_pose: np.ndarray) -> bool:
        """
        Full 6-DoF Visual-Inertial pose update (Position + Quaternion attitude).
        """
        z_pos = np.asarray(z_pos, dtype=np.float64).flatten()
        z_q = quat_normalize(np.asarray(z_q, dtype=np.float64).flatten())
        R_pose = np.asarray(R_pose, dtype=np.float64)
        
        # Position residual
        r_p = z_pos - self.state.p
        
        # Orientation residual: delta_q = z_q * q_est^-1
        # q_inv = [w, -x, -y, -z]
        q_inv = np.array([self.state.q[0], -self.state.q[1], -self.state.q[2], -self.state.q[3]], dtype=np.float64)
        delta_q = quat_multiply(z_q, q_inv)
        # Small angle approximation for rotation vector: 2 * [dx, dy, dz] * sign(dw)
        sign_w = 1.0 if delta_q[0] >= 0 else -1.0
        r_theta = 2.0 * sign_w * delta_q[1:4]
        
        residual = np.concatenate([r_p, r_theta])
        
        H = np.zeros((6, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)
        
        S = H @ self.state.P @ H.T + R_pose
        try:
            S_inv = np.linalg.inv(S)
            d_mahalanobis_sq = float(residual.T @ S_inv @ residual)
        except np.linalg.LinAlgError:
            return False
            
        self.last_mahalanobis_dist = float(np.sqrt(max(0.0, d_mahalanobis_sq)))
        
        if d_mahalanobis_sq > self.config.chi2_gate_pose:
            self.gated_measurements_count += 1
            return False
            
        K = self.state.P @ H.T @ S_inv
        delta_x = K @ residual
        self._inject_error_and_reset(delta_x, K, H, R_pose)
        self.accepted_measurements_count += 1
        return True

    def _inject_error_and_reset(self, delta_x: np.ndarray, K: np.ndarray, H: np.ndarray, R_meas: np.ndarray):
        """
        Injects estimated error state delta_x into nominal states and resets error state.
        Uses Joseph form covariance update for numerical stability.
        """
        dp = delta_x[0:3]
        dv = delta_x[3:6]
        dtheta = delta_x[6:9]
        dba = delta_x[9:12]
        dbg = delta_x[12:15]
        
        self.state.p += dp
        self.state.v += dv
        self.state.ba += dba
        self.state.bg += dbg
        
        dq = rot_vec_to_quat(dtheta)
        self.state.q = quat_normalize(quat_multiply(self.state.q, dq))
        
        I_KH = np.eye(15, dtype=np.float64) - K @ H
        self.state.P = I_KH @ self.state.P @ I_KH.T + K @ R_meas @ K.T
        
        G = np.eye(15, dtype=np.float64)
        G[6:9, 6:9] = np.eye(3) - 0.5 * skew_symmetric(dtheta)
        self.state.P = G @ self.state.P @ G.T
        self.state.P = 0.5 * (self.state.P + self.state.P.T)
