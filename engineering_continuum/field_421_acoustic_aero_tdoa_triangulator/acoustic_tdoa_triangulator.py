"""Course 421: Acoustic Engine Ingress Classifier & Time-Difference-of-Arrival (TDoA) Triangulator"""
import numpy as np

class AcousticAeroTDoATriangulator:
    def __init__(self, speed_of_sound_m_s: float = 343.0, target_harmonic_hz: float = 110.0):
        self.c = speed_of_sound_m_s
        self.target_harmonic = target_harmonic_hz # Signature two-stroke engine RPM fundamental
        self.sensor_locations = np.array([
            [0.0, 0.0, 0.0],       # Sensor A
            [1000.0, 0.0, 0.0],    # Sensor B
            [500.0, 866.0, 0.0]    # Sensor C (Equilateral triangle 1km baseline)
        ])

    def detect_engine_harmonic_signature(self, audio_signal: np.ndarray, sample_rate: float = 8000.0) -> tuple:
        """Computes Welch FFT peak around target drone engine frequency (80-160Hz)."""
        freqs = np.fft.rfftfreq(len(audio_signal), d=1.0 / sample_rate)
        fft_mag = np.abs(np.fft.rfft(audio_signal))
        
        target_mask = (freqs >= self.target_harmonic - 15.0) & (freqs <= self.target_harmonic + 15.0)
        baseline_mask = (freqs >= 40.0) & (freqs <= 400.0) & ~target_mask
        
        target_peak = float(np.max(fft_mag[target_mask])) if np.any(target_mask) else 0.0
        baseline_noise = float(np.mean(fft_mag[baseline_mask])) + 1e-6
        snr = target_peak / baseline_noise
        
        detected = bool(snr > 3.0)
        return detected, snr

    def triangulate_2d_tdoa(self, tdoas_seconds: list) -> np.ndarray:
        """
        Hyperbolic multilateration from 3 receiver nodes.
        tdoas_seconds: [t_B - t_A, t_C - t_A]
        Returns estimated (x, y) coordinates of the acoustic source.
        """
        tau_ba = tdoas_seconds[0]
        tau_ca = tdoas_seconds[1]
        
        d_ba = tau_ba * self.c
        d_ca = tau_ca * self.c
        
        # Linearized approximation around sensor array centroid
        # [x_B - x_A, y_B - y_A] * [x; y] ~ approx
        s_a = self.sensor_locations[0][:2]
        s_b = self.sensor_locations[1][:2]
        s_c = self.sensor_locations[2][:2]
        
        # Grid search over 2km bounding area for minimal hyperbolic residual
        best_pos = np.array([500.0, 300.0])
        min_residual = float("inf")
        
        x_grid = np.linspace(-500.0, 1500.0, 40)
        y_grid = np.linspace(-500.0, 1500.0, 40)
        
        for x in x_grid:
            for y in y_grid:
                pt = np.array([x, y])
                dist_a = np.linalg.norm(pt - s_a)
                dist_b = np.linalg.norm(pt - s_b)
                dist_c = np.linalg.norm(pt - s_c)
                
                err_b = (dist_b - dist_a) - d_ba
                err_c = (dist_c - dist_a) - d_ca
                res = err_b ** 2 + err_c ** 2
                
                if res < min_residual:
                    min_residual = res
                    best_pos = pt
                    
        return best_pos

    def verify_properties(self) -> bool:
        t = np.arange(8000) / 8000.0
        synth = np.sin(2 * np.pi * 110.0 * t) * 5.0 + np.random.normal(0, 0.1, 8000)
        det, snr = self.detect_engine_harmonic_signature(synth)
        return bool(det and snr > 3.0)
