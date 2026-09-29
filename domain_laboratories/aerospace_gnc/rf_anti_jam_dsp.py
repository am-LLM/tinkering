"""
RF Anti-Jamming DSP: Real-Time Adaptive IIR Notch Filter & Spatial Null-Steering Beamformer
-----------------------------------------------------------------------------------------
Provides electronic warfare (EW) anti-jamming and anti-spoofing countermeasures:
1. Real-time adaptive IIR notch filtering with gradient frequency tracking for high-power narrowband jamming cancellation (>40 dB notch depth).
2. Multi-element spatial null-steering beamforming (Robust Capon / MVDR & Frost LCMV) to place deep spatial nulls at jamming angles while preserving GNSS / satellite signal of interest (SOI).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any


# =====================================================================
# 1. Real-Time Adaptive IIR Notch Filter
# =====================================================================

class AdaptiveIIRNotchFilter:
    """
    Second-order adaptive IIR notch filter with exact gradient frequency tracking.
    Transfer Function:
        H(z) = (1 + a*z^-1 + z^-2) / (1 + rho*a*z^-1 + rho^2*z^-2)
        where a = -2*cos(w0).
    """
    def __init__(self, fs: float = 10000.0, rho: float = 0.95, mu: float = 0.05,
                 initial_f0: Optional[float] = None, adaptive: bool = True):
        """
        Args:
            fs: Sampling rate (Hz)
            rho: Pole contraction factor (0.8 < rho < 0.999).
            mu: Adaptation step size for frequency tracking.
            initial_f0: Initial notch frequency estimate (Hz).
            adaptive: Whether to continuously update coefficient 'a'.
        """
        self.fs = float(fs)
        self.rho = float(rho)
        self.mu = float(mu)
        self.adaptive = bool(adaptive)
        
        # Initialize notch center coefficient 'a'
        if initial_f0 is not None:
            w0 = 2.0 * np.pi * initial_f0 / self.fs
            self.a = float(-2.0 * np.cos(w0))
        else:
            self.a = 0.0  # corresponds to w0 = pi/2 (fs/4)
            
        # Filter internal states (delay lines)
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        
        # Exact recursive gradient filter states: g[n] = dy[n]/da
        self.g1 = 0.0
        self.g2 = 0.0

    @property
    def notch_frequency(self) -> float:
        """Returns the current estimated notch center frequency in Hz."""
        cos_w0 = -self.a / 2.0
        cos_w0 = float(np.clip(cos_w0, -0.9999, 0.9999))
        w0 = np.arccos(cos_w0)
        return float(w0 * self.fs / (2.0 * np.pi))

    def process_sample(self, x: float) -> float:
        """
        Processes a single input sample in real time, updating filter state and tracking jammer frequency.
        """
        x = float(x)
        # Direct form IIR realization:
        # y[n] = x[n] + a*x[n-1] + x[n-2] - rho*a*y[n-1] - (rho^2)*y[n-2]
        y = x + self.a * self.x1 + self.x2 - self.rho * self.a * self.y1 - (self.rho ** 2) * self.y2
        
        # Exact recursive gradient: g[n] = x[n-1] - rho*y[n-1] - rho*a*g[n-1] - rho^2*g[n-2]
        g = self.x1 - self.rho * self.y1 - self.rho * self.a * self.g1 - (self.rho ** 2) * self.g2
        
        if self.adaptive:
            # Normalized gradient descent update
            norm = (g ** 2) + 1e-4
            self.a = self.a - (self.mu * y * g) / norm
            self.a = float(np.clip(self.a, -1.98, 1.98))
        
        # Update delay lines
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        self.g2 = self.g1
        self.g1 = g
        
        return y

    def filter_signal(self, signal: np.ndarray) -> np.ndarray:
        """Processes an entire batch of samples sequentially in real-time mode."""
        signal = np.asarray(signal, dtype=np.float64)
        output = np.zeros_like(signal)
        for i in range(len(signal)):
            output[i] = self.process_sample(signal[i])
        return output

    def get_frequency_response(self, freqs: np.ndarray) -> np.ndarray:
        """Computes the complex frequency response H(e^jw) at specified frequencies."""
        w = 2.0 * np.pi * freqs / self.fs
        z_inv = np.exp(-1j * w)
        z_inv2 = np.exp(-2j * w)
        
        num = 1.0 + self.a * z_inv + z_inv2
        den = 1.0 + self.rho * self.a * z_inv + (self.rho ** 2) * z_inv2
        return num / den


class CascadedAdaptiveNotchFilter:
    """
    Cascaded multi-stage adaptive IIR notch filter for suppressing multiple simultaneous jammers.
    """
    def __init__(self, num_notches: int = 2, fs: float = 10000.0, rho: float = 0.95, mu: float = 0.05,
                 initial_f0s: Optional[List[float]] = None, adaptive: bool = True):
        self.stages = []
        for i in range(num_notches):
            init_f = initial_f0s[i] if (initial_f0s is not None and i < len(initial_f0s)) else (fs * (i + 1) / (2 * (num_notches + 1)))
            self.stages.append(AdaptiveIIRNotchFilter(fs=fs, rho=rho, mu=mu, initial_f0=init_f, adaptive=adaptive))

    def process_sample(self, x: float) -> float:
        val = x
        for stage in self.stages:
            val = stage.process_sample(val)
        return val

    def filter_signal(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float64)
        output = np.zeros_like(signal)
        for i in range(len(signal)):
            output[i] = self.process_sample(signal[i])
        return output


# =====================================================================
# 2. Spatial Null-Steering Array Beamformer
# =====================================================================

@dataclass
class ULAArrayConfig:
    num_elements: int = 8             # Number of antenna elements
    element_spacing: float = 0.5      # Normalized spacing d/lambda (typically 0.5 for half-wavelength)
    diagonal_loading: float = 1e-4    # Diagonal loading factor for robust inversion


class SpatialNullSteeringBeamformer:
    """
    Minimum Variance Distortionless Response (MVDR) and Linearly Constrained
    Minimum Variance (LCMV) Beamformer for spatial null-steering and jamming cancellation.
    """
    def __init__(self, config: Optional[ULAArrayConfig] = None):
        self.config = config or ULAArrayConfig()
        self.weights: np.ndarray = np.ones(self.config.num_elements, dtype=np.complex128) / self.config.num_elements

    def steering_vector(self, angle_deg: float) -> np.ndarray:
        """
        Computes the ULA array manifold steering vector a(theta).
        theta in degrees from broadside (-90 to +90 deg).
        """
        theta_rad = np.deg2rad(angle_deg)
        m = np.arange(self.config.num_elements, dtype=np.float64)
        phase = -2.0 * np.pi * self.config.element_spacing * m * np.sin(theta_rad)
        return np.exp(1j * phase)

    def compute_mvdr_weights(self, snapshots: np.ndarray, target_soi_deg: float) -> np.ndarray:
        """
        Computes MVDR / Robust Capon beamformer optimal weight vector w.
        Minimizes total array output power subject to w^H * a(theta_soi) = 1.
        """
        snapshots = np.asarray(snapshots, dtype=np.complex128)
        num_elements, num_samples = snapshots.shape
        assert num_elements == self.config.num_elements, f"Expected {self.config.num_elements} antenna channels, got {num_elements}"
        
        Rxx = (snapshots @ snapshots.conj().T) / num_samples
        tr = np.trace(Rxx).real / num_elements
        Rxx_loaded = Rxx + (self.config.diagonal_loading * max(tr, 1.0) * np.eye(num_elements, dtype=np.complex128))
        
        Rxx_inv = np.linalg.pinv(Rxx_loaded)
        a_soi = self.steering_vector(target_soi_deg)
        
        num = Rxx_inv @ a_soi
        den = np.vdot(a_soi, num)
        
        self.weights = num / den
        return self.weights

    def compute_lcmv_weights(self, snapshots: np.ndarray, constraint_angles_deg: List[float], constraint_responses: List[complex]) -> np.ndarray:
        """
        Linearly Constrained Minimum Variance (LCMV) Beamformer:
        Minimizes output power subject to C^H * w = f.
        """
        snapshots = np.asarray(snapshots, dtype=np.complex128)
        num_elements, num_samples = snapshots.shape
        
        Rxx = (snapshots @ snapshots.conj().T) / num_samples
        tr = np.trace(Rxx).real / num_elements
        Rxx_loaded = Rxx + (self.config.diagonal_loading * max(tr, 1.0) * np.eye(num_elements, dtype=np.complex128))
        Rxx_inv = np.linalg.pinv(Rxx_loaded)
        
        C = np.column_stack([self.steering_vector(ang) for ang in constraint_angles_deg])
        f = np.asarray(constraint_responses, dtype=np.complex128)
        
        R_inv_C = Rxx_inv @ C
        C_H_R_inv_C_inv = np.linalg.pinv(C.conj().T @ R_inv_C)
        self.weights = R_inv_C @ (C_H_R_inv_C_inv @ f)
        return self.weights

    def beamform(self, snapshots: np.ndarray) -> np.ndarray:
        """Applies current spatial weights to array snapshots: y[k] = w^H * x[k]."""
        snapshots = np.asarray(snapshots, dtype=np.complex128)
        return self.weights.conj().T @ snapshots

    def compute_beampattern(self, angles_deg: np.ndarray) -> np.ndarray:
        """Computes array beampattern power gain in dB across specified angles."""
        angles_deg = np.asarray(angles_deg, dtype=np.float64)
        gains_db = np.zeros_like(angles_deg)
        for i, ang in enumerate(angles_deg):
            a_theta = self.steering_vector(ang)
            response = np.abs(np.vdot(self.weights, a_theta))
            gains_db[i] = 20.0 * np.log10(max(response, 1e-10))
        return gains_db
