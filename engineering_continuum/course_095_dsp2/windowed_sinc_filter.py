"""Course 095: Windowed Sinc FIR Low-Pass Filter Design"""
import numpy as np

class WindowedSincFilter:
    @staticmethod
    def design_lowpass(cutoff_norm: float, num_taps: int = 31) -> np.ndarray:
        if num_taps % 2 == 0:
            num_taps += 1
        m = (num_taps - 1) // 2
        n = np.arange(-m, m + 1)
        # Sinc function
        h = 2 * cutoff_norm * np.sinc(2 * cutoff_norm * n)
        # Hamming window
        w = 0.54 + 0.46 * np.cos(np.pi * n / m)
        h_win = h * w
        return h_win / np.sum(h_win)
