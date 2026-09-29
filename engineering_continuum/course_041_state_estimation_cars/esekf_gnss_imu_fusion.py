"""Course 041: Error-State Extended Kalman Filter (ES-EKF) for Vehicle Localization"""
import numpy as np

class ESEKFFusion:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.p = np.zeros(3) # Position
        self.v = np.zeros(3) # Velocity
        self.cov = np.eye(6) * 0.01 # Error covariance (pos, vel)
        self.q = np.eye(6) * 1e-4   # Process noise
        self.r = np.eye(3) * 0.05   # GNSS measurement noise

    def predict(self, accel: np.ndarray):
        # Nominal state propagation
        self.p += self.v * self.dt + 0.5 * accel * (self.dt ** 2)
        self.v += accel * self.dt
        
        # Error state Jacobian F = [[I, I*dt], [0, I]]
        f = np.eye(6)
        f[0:3, 3:6] = np.eye(3) * self.dt
        self.cov = f @ self.cov @ f.T + self.q

    def update_gnss(self, gnss_pos: np.ndarray):
        h = np.zeros((3, 6))
        h[0:3, 0:3] = np.eye(3)
        
        innovation = gnss_pos - self.p
        s = h @ self.cov @ h.T + self.r
        k = self.cov @ h.T @ np.linalg.inv(s)
        
        error_state = k @ innovation
        self.p += error_state[0:3]
        self.v += error_state[3:6]
        self.cov = (np.eye(6) - k @ h) @ self.cov
        return self.p
