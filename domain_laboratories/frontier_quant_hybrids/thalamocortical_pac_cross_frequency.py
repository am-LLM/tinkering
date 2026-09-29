"""
Frontier Quant Hybrid Engine 4: Thalamocortical Phase-Amplitude Cross-Frequency Coupling (NEURO-COUP)
Calculates Modulation Index (MI) and Phase-Locking Values (PLV) across low-frequency macro theta waves
and high-frequency micro order book gamma bursts using the Hilbert transform.
"""
import numpy as np
from typing import Tuple, Dict, Any, List

class NeuroPACAnalyzer:
    def __init__(self, num_phase_bins: int = 18):
        self.n_bins = num_phase_bins
        self.bin_edges = np.linspace(-np.pi, np.pi, num_phase_bins + 1)

    @staticmethod
    def hilbert_analytic_signal(x: np.ndarray) -> np.ndarray:
        """Computes analytic signal using FFT Hilbert transform."""
        n = len(x)
        x_fft = np.fft.fft(x)
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = 1
            h[1:n//2] = 2
            h[n//2] = 1
        else:
            h[0] = 1
            h[1:(n+1)//2] = 2
        return np.fft.ifft(x_fft * h)

    def extract_phase_and_amplitude(self, low_freq_signal: np.ndarray, high_freq_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts instantaneous phase of low frequency and envelope amplitude of high frequency."""
        analytic_low = self.hilbert_analytic_signal(low_freq_signal)
        phase_low = np.angle(analytic_low)
        
        analytic_high = self.hilbert_analytic_signal(high_freq_signal)
        amp_high = np.abs(analytic_high)
        return phase_low, amp_high

    def compute_modulation_index(self, phase_low: np.ndarray, amp_high: np.ndarray) -> float:
        """Tort et al. Modulation Index MI = D_KL(P, U) / log(N_bins)"""
        bin_indices = np.digitize(phase_low, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        mean_amp = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            mask = (bin_indices == i)
            if np.any(mask):
                mean_amp[i] = np.mean(amp_high[mask])
            else:
                mean_amp[i] = 1e-12
                
        # Normalize to probability distribution P
        p = mean_amp / np.sum(mean_amp)
        # Uniform distribution U
        u = 1.0 / self.n_bins
        # Kullback-Leibler divergence
        d_kl = np.sum(p * np.log(p / u))
        mi = d_kl / np.log(self.n_bins)
        return float(mi)

    def phase_locking_value(self, phase_series_a: np.ndarray, phase_series_b: np.ndarray) -> float:
        """PLV = | 1/N sum exp(i * (phi_a - phi_b)) |"""
        delta_phase = phase_series_a - phase_series_b
        complex_sum = np.sum(np.exp(1j * delta_phase)) / len(delta_phase)
        return float(np.abs(complex_sum))
