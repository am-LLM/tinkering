"""Engine 49: Jeffress Coincidence Delay-Line + 3D Binaural Sound Localizer."""
from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple

class JeffressSoundLocalizerEngine:
    def __init__(self, sample_rate_hz: int = 44100, head_radius_m: float = 0.0875, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.fs = sample_rate_hz
        self.head_radius = head_radius_m # Woodworth head shadow model
        self.c_sound = 343.0 # m/s
        self.max_itd_samples = int(np.ceil((2 * head_radius_m / self.c_sound) * sample_rate_hz))
        self.num_coincidence_detectors = 2 * self.max_itd_samples + 1

    def compute_binaural_signals(self, true_azimuth_deg: float, freq_hz: float = 500.0, duration_sec: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Generate left and right ear signals using Woodworth ITD formula: ITD = (r/c) * (theta + sin(theta))."""
        t = np.linspace(0, duration_sec, int(self.fs * duration_sec))
        rad = np.radians(true_azimuth_deg)
        itd_sec = (self.head_radius / self.c_sound) * (rad + np.sin(rad))
        
        left_signal = np.sin(2 * np.pi * freq_hz * t)
        right_signal = np.sin(2 * np.pi * freq_hz * (t - itd_sec)) + self.rng.normal(0, 0.02, size=len(t))
        return left_signal, right_signal

    def localize_sound_azimuth(self, left_sig: np.ndarray, right_sig: np.ndarray) -> Dict[str, float]:
        """Jeffress cross-correlation delay-line coincidence map."""
        cross_corr = np.correlate(left_sig, right_sig, mode='full')
        mid = len(cross_corr) // 2
        lags = np.arange(-self.max_itd_samples, self.max_itd_samples + 1)
        
        subset = cross_corr[mid - self.max_itd_samples : mid + self.max_itd_samples + 1]
        best_lag_samples = lags[np.argmax(subset)]
        measured_itd_sec = -best_lag_samples / self.fs
        
        # Invert Woodworth formula for approximate azimuth
        estimated_azimuth_deg = float(np.degrees(measured_itd_sec * self.c_sound / (2 * self.head_radius)))
        estimated_azimuth_deg = float(np.clip(estimated_azimuth_deg, -90.0, 90.0))

        return {
            "estimated_azimuth_deg": estimated_azimuth_deg,
            "measured_itd_ms": float(measured_itd_sec * 1e3),
            "coincidence_peak_strength": float(np.max(subset))
        }
