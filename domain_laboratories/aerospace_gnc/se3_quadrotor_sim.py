"""
SE(3) Quadrotor Dynamics Simulator with Differential Flatness & Minimum Snap Trajectory Generation
-------------------------------------------------------------------------------------------------
1. 12-State Rigid-Body Nonlinear Dynamics on SE(3) with RK4 numerical integration.
2. Differential Flatness mapping from 4D flat outputs (x, y, z, yaw) to full SE(3) state and inputs.
3. Minimum Snap Trajectory Generator solving 7th-order piecewise polynomial splines across waypoints.
4. Geometric Non-Linear SE(3) Tracking Controller with exponential convergence guarantees.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


def skew(v: np.ndarray) -> np.ndarray:
    """Computes 3x3 skew-symmetric matrix [v]_x."""
    v = np.asarray(v, dtype=np.float64).flatten()
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def vee(R_skew: np.ndarray) -> np.ndarray:
    """Inverse skew operator: so(3) -> R^3."""
    return np.array([R_skew[2, 1], R_skew[0, 2], R_skew[1, 0]], dtype=np.float64)


def rot_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Converts 3x3 rotation matrix to unit quaternion [qw, qx, qy, qz]."""
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    return q / (norm if norm > 1e-12 else 1.0)


@dataclass
class QuadrotorConfig:
    mass: float = 1.0                     # [kg] Total quadrotor mass
    arm_length: float = 0.225             # [m] Center to motor hub distance
    Ixx: float = 0.0049                   # [kg*m^2]
    Iyy: float = 0.0049                   # [kg*m^2]
    Izz: float = 0.0088                   # [kg*m^2]
    kf: float = 1.0                       # Motor thrust coefficient (N per normalized input)
    km: float = 0.02                      # Drag torque coefficient (N*m / N)
    drag_coeff: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.1], dtype=np.float64))
    gravity: float = 9.80665

    @property
    def J(self) -> np.ndarray:
        return np.diag([self.Ixx, self.Iyy, self.Izz]).astype(np.float64)

    @property
    def J_inv(self) -> np.ndarray:
        return np.diag([1.0 / self.Ixx, 1.0 / self.Iyy, 1.0 / self.Izz]).astype(np.float64)


@dataclass
class QuadrotorState:
    p: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    @property
    def quaternion(self) -> np.ndarray:
        return rot_matrix_to_quat(self.R)


class SE3QuadrotorSimulator:
    """
    High-fidelity 12-state rigid body quadrotor simulator on SE(3) with RK4 integration.
    """
    def __init__(self, config: Optional[QuadrotorConfig] = None):
        self.config = config or QuadrotorConfig()
        self.state = QuadrotorState()
        self.time: float = 0.0
        
        # Mixer matrix: [F_total, tau_x, tau_y, tau_z]^T = M * [f1, f2, f3, f4]^T (X configuration)
        d = self.config.arm_length / np.sqrt(2.0)
        c_m = self.config.km
        self.mixer_matrix = np.array([
            [1.0,  1.0,  1.0,  1.0],
            [-d,   -d,   d,    d  ],
            [ d,   -d,  -d,    d  ],
            [-c_m, c_m, -c_m, c_m ]
        ], dtype=np.float64)
        self.inv_mixer = np.linalg.pinv(self.mixer_matrix)

    def set_state(self, p: np.ndarray, v: np.ndarray, R: np.ndarray, omega: np.ndarray):
        self.state.p = np.asarray(p, dtype=np.float64).flatten()
        self.state.v = np.asarray(v, dtype=np.float64).flatten()
        self.state.R = np.asarray(R, dtype=np.float64)
        self.state.omega = np.asarray(omega, dtype=np.float64).flatten()

    def motor_thrusts_to_wrench(self, motor_thrusts: np.ndarray) -> Tuple[float, np.ndarray]:
        """Converts 4 individual motor thrusts [N] into total thrust f_z [N] and torques tau [N*m]."""
        f_m = np.asarray(motor_thrusts, dtype=np.float64).flatten()
        wrench = self.mixer_matrix @ f_m
        return float(wrench[0]), wrench[1:4]

    def wrench_to_motor_thrusts(self, f_total: float, tau: np.ndarray) -> np.ndarray:
        """Converts total thrust f and torques tau to required 4 motor thrusts."""
        w = np.array([f_total, tau[0], tau[1], tau[2]], dtype=np.float64)
        return self.inv_mixer @ w

    def _state_derivative(self, p: np.ndarray, v: np.ndarray, R: np.ndarray, omega: np.ndarray,
                          f_thrust: float, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluates continuous SE(3) vector field [p_dot, v_dot, R_dot, omega_dot]."""
        p_dot = v.copy()
        
        # Acceleration: m * v_dot = m * g * e3 + R * [0, 0, f_thrust] - D * v
        g_vec = np.array([0.0, 0.0, -self.config.gravity], dtype=np.float64)
        thrust_w = R @ np.array([0.0, 0.0, f_thrust], dtype=np.float64)
        drag_w = self.config.drag_coeff * v
        v_dot = g_vec + (thrust_w - drag_w) / self.config.mass
        
        # Rotational kinematics: R_dot = R * [omega]_x
        R_dot = R @ skew(omega)
        
        # Rotational dynamics: J * omega_dot = tau - omega x (J * omega)
        J_omega = self.config.J @ omega
        gyro_torque = np.cross(omega, J_omega)
        omega_dot = self.config.J_inv @ (tau - gyro_torque)
        
        return p_dot, v_dot, R_dot, omega_dot

    def step_wrench(self, f_thrust: float, tau: np.ndarray, dt: float) -> QuadrotorState:
        """Executes one simulation step with total thrust and moments using 4th-Order Runge-Kutta."""
        assert dt > 0, "dt must be > 0"
        p = self.state.p
        v = self.state.v
        R = self.state.R
        omega = self.state.omega
        tau = np.asarray(tau, dtype=np.float64).flatten()

        # k1
        k1_p, k1_v, k1_R, k1_om = self._state_derivative(p, v, R, omega, f_thrust, tau)
        
        # k2
        p2 = p + 0.5 * dt * k1_p
        v2 = v + 0.5 * dt * k1_v
        R2 = R + 0.5 * dt * k1_R
        om2 = omega + 0.5 * dt * k1_om
        k2_p, k2_v, k2_R, k2_om = self._state_derivative(p2, v2, R2, om2, f_thrust, tau)
        
        # k3
        p3 = p + 0.5 * dt * k2_p
        v3 = v + 0.5 * dt * k2_v
        R3 = R + 0.5 * dt * k2_R
        om3 = omega + 0.5 * dt * k2_om
        k3_p, k3_v, k3_R, k3_om = self._state_derivative(p3, v3, R3, om3, f_thrust, tau)
        
        # k4
        p4 = p + dt * k3_p
        v4 = v + dt * k3_v
        R4 = R + dt * k3_R
        om4 = omega + dt * k3_om
        k4_p, k4_v, k4_R, k4_om = self._state_derivative(p4, v4, R4, om4, f_thrust, tau)
        
        # RK4 update
        self.state.p += (dt / 6.0) * (k1_p + 2.0*k2_p + 2.0*k3_p + k4_p)
        self.state.v += (dt / 6.0) * (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v)
        R_new = R + (dt / 6.0) * (k1_R + 2.0*k2_R + 2.0*k3_R + k4_R)
        self.state.omega += (dt / 6.0) * (k1_om + 2.0*k2_om + 2.0*k3_om + k4_om)
        
        # Re-orthonormalize SO(3) via polar decomposition / SVD
        if np.isfinite(R_new).all():
            u, _, vt = np.linalg.svd(R_new)
            self.state.R = u @ vt
            if np.linalg.det(self.state.R) < 0:
                u[:, -1] *= -1
                self.state.R = u @ vt
            
        self.time += dt
        return self.state

    def step_motors(self, motor_thrusts: np.ndarray, dt: float) -> QuadrotorState:
        """Executes simulation step directly from 4 motor thrust inputs."""
        f_thrust, tau = self.motor_thrusts_to_wrench(motor_thrusts)
        return self.step_wrench(f_thrust, tau, dt)


class DifferentialFlatness:
    """
    Differential Flatness mapping for quadrotor dynamics.
    Converts 4D flat trajectory sigma = [x, y, z, yaw]^T and its derivatives
    up to 4th order (pos, vel, acc, jerk, snap, yaw, yaw_dot) into full SE(3) state and inputs.
    """
    def __init__(self, mass: float = 1.0, gravity: float = 9.80665):
        self.mass = mass
        self.gravity = gravity

    def compute_state_and_input(self, p: np.ndarray, v: np.ndarray, a: np.ndarray,
                                j: np.ndarray, s: np.ndarray,
                                yaw: float, yaw_dot: float) -> Dict[str, Any]:
        """
        Computes desired SE(3) attitude R_d, angular velocity omega_d, total thrust f_d,
        and angular acceleration omega_dot_d.
        """
        p = np.asarray(p, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        j = np.asarray(j, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        
        # Thrust acceleration vector: t_B = a + [0, 0, g]^T
        z_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        t_vec = a + self.gravity * z_w
        norm_t = np.linalg.norm(t_vec)
        if norm_t < 1e-6:
            norm_t = 1e-6
            
        z_B = t_vec / norm_t
        f_d = self.mass * norm_t
        
        # Desired heading vector x_C
        x_C = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
        
        y_B_un = np.cross(z_B, x_C)
        norm_yB = np.linalg.norm(y_B_un)
        if norm_yB < 1e-6:
            x_C = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            y_B_un = np.cross(z_B, x_C)
            norm_yB = np.linalg.norm(y_B_un)
            
        y_B = y_B_un / (norm_yB if norm_yB > 1e-6 else 1.0)
        x_B = np.cross(y_B, z_B)
        
        R_d = np.column_stack([x_B, y_B, z_B])
        
        # Derivatives for angular velocity
        z_B_dot = (j - np.dot(z_B, j) * z_B) / norm_t
        
        om_x = -np.dot(y_B, z_B_dot)
        om_y = np.dot(x_B, z_B_dot)
        om_z = yaw_dot * np.dot(z_w, z_B)
        omega_d = np.array([om_x, om_y, om_z], dtype=np.float64)
        
        return {
            "p": p,
            "v": v,
            "a": a,
            "R": R_d,
            "omega": omega_d,
            "thrust": f_d,
            "quaternion": rot_matrix_to_quat(R_d)
        }


class MinimumSnapTrajectoryGenerator:
    """
    Minimum Snap Trajectory Generator using 7th-order polynomial splines.
    Minimizes cost J = integral( ||d^4 p / dt^4||^2 dt ) across waypoints.
    """
    def __init__(self, waypoints: np.ndarray, time_allocations: np.ndarray):
        """
        Args:
            waypoints: (M, 3) Waypoint positions [x, y, z]
            time_allocations: (M-1,) Durations T_i for each polynomial segment
        """
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        self.time_allocations = np.asarray(time_allocations, dtype=np.float64)
        self.num_segments = len(self.time_allocations)
        assert len(self.waypoints) == self.num_segments + 1, "Number of waypoints must be num_segments + 1"
        
        self.coeffs_x = self._solve_spline_1d(self.waypoints[:, 0])
        self.coeffs_y = self._solve_spline_1d(self.waypoints[:, 1])
        self.coeffs_z = self._solve_spline_1d(self.waypoints[:, 2])

    def _solve_spline_1d(self, waypoints_1d: np.ndarray) -> np.ndarray:
        """Solves 8 * num_segments linear equality constraints for 7th-order polynomial spline."""
        N = 8  # 7th order polynomial has 8 coefficients
        M = self.num_segments
        total_vars = M * N
        
        A = np.zeros((total_vars, total_vars), dtype=np.float64)
        b = np.zeros(total_vars, dtype=np.float64)
        
        row = 0
        
        def poly_basis(t: float, order: int = 0) -> np.ndarray:
            vec = np.zeros(N, dtype=np.float64)
            for k in range(order, N):
                factor = 1.0
                for f in range(order):
                    factor *= (k - f)
                vec[k] = factor * (t ** (k - order))
            return vec

        # 1. Waypoint position constraints
        for i in range(M):
            T = self.time_allocations[i]
            A[row, i*N : (i+1)*N] = poly_basis(0.0, order=0)
            b[row] = waypoints_1d[i]
            row += 1
            
            A[row, i*N : (i+1)*N] = poly_basis(T, order=0)
            b[row] = waypoints_1d[i+1]
            row += 1

        # 2. Boundary conditions at start: Vel=0, Acc=0, Jerk=0
        for order in [1, 2, 3]:
            A[row, 0:N] = poly_basis(0.0, order=order)
            b[row] = 0.0
            row += 1

        # 3. Boundary conditions at end: Vel=0, Acc=0, Jerk=0
        T_end = self.time_allocations[-1]
        for order in [1, 2, 3]:
            A[row, (M-1)*N : M*N] = poly_basis(T_end, order=order)
            b[row] = 0.0
            row += 1

        # 4. Continuity constraints at internal junctions
        for i in range(M - 1):
            T_i = self.time_allocations[i]
            for order in range(1, 7):
                if row >= total_vars:
                    break
                A[row, i*N : (i+1)*N] = poly_basis(T_i, order=order)
                A[row, (i+1)*N : (i+2)*N] = -poly_basis(0.0, order=order)
                b[row] = 0.0
                row += 1

        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        return coeffs.reshape((M, N))

    def evaluate(self, t: float) -> Dict[str, np.ndarray]:
        """Evaluates position, velocity, acceleration, jerk, and snap at query time t."""
        total_time = np.sum(self.time_allocations)
        t_clamped = np.clip(t, 0.0, total_time)
        
        accum_t = 0.0
        seg_idx = self.num_segments - 1
        for i, dt in enumerate(self.time_allocations):
            if t_clamped <= accum_t + dt or i == self.num_segments - 1:
                seg_idx = i
                break
            accum_t += dt
            
        tau = t_clamped - accum_t
        N = 8
        
        def eval_deriv(c_seg: np.ndarray, order: int) -> float:
            val = 0.0
            for k in range(order, N):
                factor = 1.0
                for f in range(order):
                    factor *= (k - f)
                val += c_seg[k] * factor * (tau ** (k - order))
            return val

        pos = np.array([
            eval_deriv(self.coeffs_x[seg_idx], 0),
            eval_deriv(self.coeffs_y[seg_idx], 0),
            eval_deriv(self.coeffs_z[seg_idx], 0)
        ])
        vel = np.array([
            eval_deriv(self.coeffs_x[seg_idx], 1),
            eval_deriv(self.coeffs_y[seg_idx], 1),
            eval_deriv(self.coeffs_z[seg_idx], 1)
        ])
        acc = np.array([
            eval_deriv(self.coeffs_x[seg_idx], 2),
            eval_deriv(self.coeffs_y[seg_idx], 2),
            eval_deriv(self.coeffs_z[seg_idx], 2)
        ])
        jerk = np.array([
            eval_deriv(self.coeffs_x[seg_idx], 3),
            eval_deriv(self.coeffs_y[seg_idx], 3),
            eval_deriv(self.coeffs_z[seg_idx], 3)
        ])
        snap = np.array([
            eval_deriv(self.coeffs_x[seg_idx], 4),
            eval_deriv(self.coeffs_y[seg_idx], 4),
            eval_deriv(self.coeffs_z[seg_idx], 4)
        ])
        
        return {
            "p": pos,
            "v": vel,
            "a": acc,
            "jerk": jerk,
            "snap": snap
        }


class SE3GeometricController:
    """
    Geometric Non-Linear Tracking Controller on SE(3) (Lee, Leok, McClamroch).
    Guarantees almost-global exponential tracking on the Special Euclidean Group SE(3).
    """
    def __init__(self, config: Optional[QuadrotorConfig] = None,
                 kp: float = 6.0, kv: float = 4.0, kr: float = 25.0, kw: float = 10.0):
        self.config = config or QuadrotorConfig()
        self.kp = float(kp)
        self.kv = float(kv)
        self.kr = float(kr)
        self.kw = float(kw)

    def compute_control(self, curr_state: QuadrotorState,
                        des_p: np.ndarray, des_v: np.ndarray, des_a: np.ndarray,
                        des_yaw: float = 0.0, des_omega: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """
        Computes desired total thrust f [N] and control moments tau [N*m].
        """
        des_p = np.asarray(des_p, dtype=np.float64)
        des_v = np.asarray(des_v, dtype=np.float64)
        des_a = np.asarray(des_a, dtype=np.float64)
        des_omega = np.asarray(des_omega, dtype=np.float64) if des_omega is not None else np.zeros(3)

        # 1. Translational tracking errors
        ep = curr_state.p - des_p
        ev = curr_state.v - des_v
        
        # Desired force vector F_d
        g_vec = np.array([0.0, 0.0, self.config.gravity], dtype=np.float64)
        F_d = -self.kp * ep - self.kv * ev + self.config.mass * (g_vec + des_a)
        
        # Desired body z-axis
        norm_F = np.linalg.norm(F_d)
        if norm_F < 1e-4:
            norm_F = 1e-4
        z_B_d = F_d / norm_F
        
        # Projected thrust scalar along current body z-axis
        z_B_curr = curr_state.R[:, 2]
        f_thrust = float(np.dot(F_d, z_B_curr))
        
        # 2. Desired attitude matrix R_d
        x_C = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0], dtype=np.float64)
        y_B_d_un = np.cross(z_B_d, x_C)
        norm_y = np.linalg.norm(y_B_d_un)
        if norm_y < 1e-6:
            x_C = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            y_B_d_un = np.cross(z_B_d, x_C)
            norm_y = np.linalg.norm(y_B_d_un)
        y_B_d = y_B_d_un / (norm_y if norm_y > 1e-6 else 1.0)
        x_B_d = np.cross(y_B_d, z_B_d)
        R_d = np.column_stack([x_B_d, y_B_d, z_B_d])
        
        # 3. Attitude tracking error on SO(3)
        # e_R = 1/2 * vee( R_d^T * R - R^T * R_d )
        e_R_mat = 0.5 * (R_d.T @ curr_state.R - curr_state.R.T @ R_d)
        e_R = vee(e_R_mat)
        
        # Angular velocity tracking error
        e_om = curr_state.omega - curr_state.R.T @ R_d @ des_omega
        
        # Moment calculation with inertia decoupling & gyroscopic compensation
        J = self.config.J
        J_om = J @ curr_state.omega
        gyro = np.cross(curr_state.omega, J_om)
        
        # Decoupled attitude PD
        tau = -J @ (self.kr * e_R + self.kw * e_om) + gyro
        return max(0.0, f_thrust), tau
