"""Course 424: 15-State Error-State Kalman Filter (ESKF) for Visual-Inertial Odometry"""
import numpy as np

class ESKF15StateVisualInertialOdometry:
    def __init__(self, dt: float = 0.01):
        self.dt = dt
        # Nominal State: [pos(3), vel(3), quat(4), accel_bias(3), gyro_bias(3)] = 16 components
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0]) # [qw, qx, qy, qz]
        self.acc_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        
        # 15x15 Error-State Covariance Matrix
        self.P = np.eye(15) * 0.01

    def quat_to_rot_matrix(self, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])

    def predict_imu_step(self, raw_accel: np.ndarray, raw_gyro: np.ndarray) -> None:
        """Propagates nominal states via high-speed IMU integration."""
        # Correct with biases
        acc_corr = raw_accel - self.acc_bias
        gyro_corr = raw_gyro - self.gyro_bias
        
        R = self.quat_to_rot_matrix(self.quat)
        # Acceleration in world frame (subtract gravity [0, 0, 9.81])
        acc_world = np.dot(R, acc_corr) - np.array([0.0, 0.0, 9.81])
        
        self.pos += self.vel * self.dt + 0.5 * acc_world * (self.dt ** 2)
        self.vel += acc_world * self.dt
        
        # Quaternion integration
        w_norm = np.linalg.norm(gyro_corr)
        if w_norm > 1e-6:
            half_theta = 0.5 * w_norm * self.dt
            dq = np.concatenate([[np.cos(half_theta)], (gyro_corr / w_norm) * np.sin(half_theta)])
            # Quat multiply
            qw, qx, qy, qz = self.quat
            rw, rx, ry, rz = dq
            self.quat = np.array([
                qw*rw - qx*rx - qy*ry - qz*rz,
                qw*rx + qx*rw + qy*rz - qz*ry,
                qw*ry - qx*rz + qy*rw + qz*rx,
                qw*rz + qx*ry - qy*rx + qz*rw
            ])
            self.quat /= np.linalg.norm(self.quat)

    def update_optical_flow_velocity(self, optical_vel_measured: np.ndarray) -> None:
        """ESKF measurement update from downward optical flow sensor [vx, vy, vz]."""
        # Measurement matrix H: maps 15-state error to 3-state velocity
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3) # Velocity error indices
        
        R_meas = np.eye(3) * 0.05 # Optical flow noise variance
        
        # Innovation y = z - H*x_nom
        y = optical_vel_measured - self.vel
        
        # Kalman Gain K = P * H^T * (H * P * H^T + R)^-1
        S = np.dot(H, np.dot(self.P, H.T)) + R_meas
        K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))
        
        # Error state correction delta_x
        delta_x = np.dot(K, y)
        
        # Inject error corrections into nominal state
        self.pos += delta_x[0:3]
        self.vel += delta_x[3:6]
        self.acc_bias += delta_x[9:12]
        self.gyro_bias += delta_x[12:15]
        
        # Covariance update P = (I - K*H) * P
        I_KH = np.eye(15) - np.dot(K, H)
        self.P = np.dot(I_KH, self.P)

    def verify_properties(self) -> bool:
        # Stationary drone with zero motion on fresh filter
        fresh = ESKF15StateVisualInertialOdometry()
        acc = np.array([0.0, 0.0, 9.81]) # Gravitational reaction
        gyro = np.array([0.0, 0.0, 0.0])
        fresh.predict_imu_step(acc, gyro)
        fresh.update_optical_flow_velocity(np.zeros(3))
        return bool(np.all(np.isfinite(fresh.pos)) and np.linalg.norm(fresh.vel) < 0.1)
