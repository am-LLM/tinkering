"""
Zero-RF Subterranean SLAM Navigation Engine
===========================================
Integrates:
1. 3-Axis Fluxgate Crustal Anomaly Gradient Tensor Navigation (Zero-RF Geolocation).
2. 8-Channel Ultrasonic Phased Array Beamforming (Delay-and-Sum Acoustic Sonar).
3. 6-DOF Extended Kalman Filter (EKF) Subterranean Fusion SLAM Solver.
4. Cavern Occupancy Mapping & Obstacle Spatial Point Cloud Reconstruction.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


@dataclass
class CrustalMagneticMap:
    """Reference model of subterranean crustal magnetic field variations."""
    base_field_nt: np.ndarray  # [Bx, By, Bz] in nT (e.g. [20000, 5000, 45000])
    anomaly_freq: float = 0.05  # Spatial frequency of crustal variations (rad/m)

    def sample_field(self, position: np.ndarray) -> np.ndarray:
        """Returns magnetic flux density B(r) = B_0 + Delta B(r)."""
        x, y, z = position[0], position[1], position[2]
        # Multi-scale sinusoidal crustal anomaly simulation
        dx = 500.0 * math.sin(self.anomaly_freq * x) + 200.0 * math.cos(0.02 * y)
        dy = 400.0 * math.cos(self.anomaly_freq * y) + 150.0 * math.sin(0.03 * z)
        dz = 600.0 * math.sin(self.anomaly_freq * z + 0.1 * x)
        return self.base_field_nt + np.array([dx, dy, dz], dtype=np.float64)

    def compute_gradient_tensor(self, position: np.ndarray, delta: float = 0.1) -> np.ndarray:
        """Computes 3x3 spatial gradient tensor dB_i / dr_j via finite differences."""
        grad = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            pos_plus = position.copy()
            pos_minus = position.copy()
            pos_plus[j] += delta
            pos_minus[j] -= delta
            b_plus = self.sample_field(pos_plus)
            b_minus = self.sample_field(pos_minus)
            grad[:, j] = (b_plus - b_minus) / (2.0 * delta)
        return grad


class UltrasonicPhasedArray:
    """
    8-Channel Ultrasonic Phased Array Acoustic Beamformer.
    Carrier frequency: 40 kHz, Speed of sound in cave air: 343 m/s.
    """

    def __init__(
        self,
        num_channels: int = 8,
        carrier_freq_hz: float = 40000.0,
        speed_of_sound_m_s: float = 343.0,
        element_spacing_m: Optional[float] = None,
        sample_rate_hz: float = 500000.0,
    ):
        self.num_channels = num_channels
        self.carrier_freq = carrier_freq_hz
        self.c = speed_of_sound_m_s
        self.wavelength = self.c / self.carrier_freq  # ~8.575 mm
        self.d = element_spacing_m if element_spacing_m is not None else (self.wavelength / 2.0)
        self.sample_rate = sample_rate_hz

        # Channel element positions along local x-axis
        self.element_positions = np.array([
            [(i - (num_channels - 1) / 2.0) * self.d, 0.0, 0.0] for i in range(num_channels)
        ], dtype=np.float64)

    def simulate_echo_signals(
        self,
        targets_local: List[Tuple[float, float, float]],
        max_range_m: float = 10.0,
        noise_std: float = 0.02,
    ) -> np.ndarray:
        """
        Simulates raw 8-channel acoustic time-domain ADC return buffers.
        Returns array of shape (num_channels, num_samples).
        """
        num_samples = int(2.0 * max_range_m / self.c * self.sample_rate)
        t = np.linspace(0, num_samples / self.sample_rate, num_samples, endpoint=False)
        signals = np.zeros((self.num_channels, num_samples), dtype=np.float64)

        for target in targets_local:
            tx, ty, tz = target
            for ch in range(self.num_channels):
                el_pos = self.element_positions[ch]
                dist = math.sqrt((tx - el_pos[0])**2 + (ty - el_pos[1])**2 + (tz - el_pos[2])**2)
                tof = 2.0 * dist / self.c  # Round trip time of flight
                sample_idx = int(tof * self.sample_rate)
                if 0 <= sample_idx < num_samples - 50:
                    pulse_len = 30
                    pulse_t = np.arange(pulse_len) / self.sample_rate
                    pulse = np.exp(-((pulse_t - 15 / self.sample_rate) ** 2) / (2 * (5 / self.sample_rate) ** 2))
                    pulse *= np.sin(2 * np.pi * self.carrier_freq * pulse_t)
                    signals[ch, sample_idx:sample_idx + pulse_len] += pulse / max(1.0, dist)

        # Add Gaussian thermal noise
        signals += np.random.normal(0.0, noise_std, signals.shape)
        return signals

    def delay_and_sum_beamform(
        self,
        signals: np.ndarray,
        azimuth_angles_deg: np.ndarray,
        max_range_m: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Delay-and-Sum (DAS) acoustic beamforming.
        Returns: (range_grid, azimuth_angles_deg, intensity_grid)
        """
        num_samples = signals.shape[1]
        time_grid = np.arange(num_samples) / self.sample_rate
        range_grid = (time_grid * self.c) / 2.0

        intensity_grid = np.zeros((len(azimuth_angles_deg), num_samples), dtype=np.float64)

        for a_idx, angle_deg in enumerate(azimuth_angles_deg):
            theta_rad = math.radians(angle_deg)
            # Channel time delays: tau_m = m * d * sin(theta) / c
            delays = (self.element_positions[:, 0] * math.sin(theta_rad)) / self.c
            delay_samples = (delays * self.sample_rate).astype(int)

            summed_signal = np.zeros(num_samples, dtype=np.float64)
            for ch in range(self.num_channels):
                ds = delay_samples[ch]
                if ds > 0:
                    summed_signal[ds:] += signals[ch, :-ds]
                elif ds < 0:
                    summed_signal[:ds] += signals[ch, -ds:]
                else:
                    summed_signal += signals[ch]

            # Envelope detection via instantaneous energy
            intensity_grid[a_idx, :] = summed_signal ** 2

        return range_grid, azimuth_angles_deg, intensity_grid

    def extract_point_cloud(
        self,
        intensity_grid: np.ndarray,
        range_grid: np.ndarray,
        azimuth_angles_deg: np.ndarray,
        threshold: float = 2.0,
    ) -> List[Tuple[float, float, float]]:
        """Extracts 3D local coordinate obstacle hits from acoustic beamforming energy."""
        points = []
        for a_idx, angle_deg in enumerate(azimuth_angles_deg):
            theta_rad = math.radians(angle_deg)
            profile = intensity_grid[a_idx, :]
            # Find local peaks above threshold
            peaks = np.where((profile[1:-1] > profile[:-2]) & (profile[1:-1] > profile[2:]) & (profile[1:-1] > threshold))[0] + 1
            for p_idx in peaks:
                r = range_grid[p_idx]
                if r > 0.2:  # Blanking distance
                    px = r * math.sin(theta_rad)
                    py = r * math.cos(theta_rad)
                    pz = 0.0
                    points.append((float(px), float(py), float(pz)))
        return points


class SubterraneanEKFSLAM:
    """
    6-DOF Subterranean Extended Kalman Filter (EKF).
    Fuses dead reckoning with 3-axis fluxgate magnetic anomaly gradients and ultrasonic point clouds.
    State: [x, y, z, vx, vy, vz] (Position & Velocity)
    """

    def __init__(self, initial_state: np.ndarray, crustal_map: CrustalMagneticMap):
        if len(initial_state) != 6:
            raise ValueError("Initial state vector must have 6 dimensions [x, y, z, vx, vy, vz]")
        self.state = initial_state.astype(np.float64)
        self.cov = np.eye(6, dtype=np.float64) * 0.1
        self.crustal_map = crustal_map
        self.map_points: List[np.ndarray] = []

    def predict(self, dt: float, accel_body: np.ndarray, process_noise_std: float = 0.05) -> None:
        """Kinematic state prediction dx/dt = v, dv/dt = a."""
        F = np.eye(6, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt

        # State transition
        self.state[0:3] += self.state[3:6] * dt + 0.5 * accel_body * (dt ** 2)
        self.state[3:6] += accel_body * dt

        # Process noise covariance
        Q = np.eye(6, dtype=np.float64) * (process_noise_std ** 2)
        self.cov = F @ self.cov @ F.T + Q

    def update_magnetic_anomaly(self, measured_field_nt: np.ndarray, measurement_noise_std: float = 5.0) -> None:
        """Updates state using 3-axis fluxgate crustal magnetic anomaly observations."""
        pos = self.state[0:3]
        expected_field = self.crustal_map.sample_field(pos)
        grad_tensor = self.crustal_map.compute_gradient_tensor(pos)

        # Measurement Jacobian H = [dB/dr (3x3), 0 (3x3)]
        H = np.zeros((3, 6), dtype=np.float64)
        H[:, 0:3] = grad_tensor

        R = np.eye(3, dtype=np.float64) * (measurement_noise_std ** 2)
        y = measured_field_nt - expected_field  # Innovation

        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)

        self.state += K @ y
        I = np.eye(6, dtype=np.float64)
        self.cov = (I - K @ H) @ self.cov

    def update_ultrasonic_points(self, points_local: List[Tuple[float, float, float]]) -> None:
        """Transforms local acoustic detections to world frame and stores in subterranean map."""
        pos = self.state[0:3]
        for p in points_local:
            p_world = pos + np.array(p, dtype=np.float64)
            self.map_points.append(p_world)
