"""Course 097: Polyphase Multirate Decimator and Anti-Aliasing DSP"""
import numpy as np

class PolyphaseDecimator:
    def __init__(self, decimation_factor: int, filter_taps: np.ndarray):
        self.m = decimation_factor
        self.taps = filter_taps

    def process(self, signal: np.ndarray) -> np.ndarray:
        filtered = np.convolve(signal, self.taps, mode='same')
        return filtered[::self.m]
