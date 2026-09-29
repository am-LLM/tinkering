"""Course 377: IEEE C37.118 Synchrophasor Estimation using Dynamic DFT Filterbank"""
import numpy as np

class SynchrophasorDFTEstimator:
    def __init__(self, nominal_freq: float = 60.0, samples_per_cycle: int = 48):
        self.f0 = nominal_freq
        self.n = samples_per_cycle
        self.fs = nominal_freq * samples_per_cycle
        self.window = np.hanning(self.n)

    def estimate_phasor(self, wave_segment: np.ndarray) -> tuple[float, float, float]:
        # returns (rms_magnitude, phase_radians, frequency_hz)
        assert len(wave_segment) == self.n
        windowed = wave_segment * self.window
        dft_bin = np.sum(windowed * np.exp(-1j * 2.0 * np.pi * np.arange(self.n) / self.n))
        # Normalization for Hanning window peak
        mag = (np.abs(dft_bin) * 2.0 / np.sum(self.window)) / np.sqrt(2.0)
        phase = np.angle(dft_bin)
        return float(mag), float(phase), self.f0
