"""Course 027: Radix-2 Decimation-in-Time Fast Fourier Transform"""
import numpy as np

def radix2_fft(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n <= 1:
        return x
    even = radix2_fft(x[0::2])
    odd = radix2_fft(x[1::2])
    terms = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    return np.concatenate([even + terms * odd, even - terms * odd])
