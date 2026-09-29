"""Course 094: Radix-2 Cooley-Tukey Discrete Fast Fourier Transform (FFT)"""
import numpy as np

class CooleyTukeyFFT:
    @staticmethod
    def fft(x: np.ndarray) -> np.ndarray:
        n = len(x)
        if n <= 1:
            return x.astype(np.complex128)
        if n % 2 != 0:
            raise ValueError("Input length must be a power of 2")
        even = CooleyTukeyFFT.fft(x[0::2])
        odd = CooleyTukeyFFT.fft(x[1::2])
        twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n)
        return np.concatenate([even + twiddle * odd, even - twiddle * odd])
