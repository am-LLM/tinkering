"""Course 247: Nitrogen-Vacancy (NV) Center ODMR Magnetometry Solver"""
import numpy as np

class NVCenterMagnetometer:
    GYROMAGNETIC_RATIO_GHZ_T = 28.0 # GHz / Tesla
    ZERO_FIELD_SPLITTING_D_GHZ = 2.87 # GHz

    def __init__(self, linewidth_mhz: float = 5.0):
        self.gamma = self.GYROMAGNETIC_RATIO_GHZ_T
        self.d_zero = self.ZERO_FIELD_SPLITTING_D_GHZ
        self.gamma_fwhm = linewidth_mhz * 1e-3 # In GHz

    def resonance_frequencies(self, b_z_tesla: float) -> tuple[float, float]:
        f_minus = self.d_zero - self.gamma * b_z_tesla
        f_plus = self.d_zero + self.gamma * b_z_tesla
        return float(f_minus), float(f_plus)

    def odmr_spectrum(self, mw_freqs_ghz: np.ndarray, b_z_tesla: float, contrast: float = 0.1) -> np.ndarray:
        f_m, f_p = self.resonance_frequencies(b_z_tesla)
        lor_m = (self.gamma_fwhm / 2.0)**2 / ((mw_freqs_ghz - f_m)**2 + (self.gamma_fwhm / 2.0)**2)
        lor_p = (self.gamma_fwhm / 2.0)**2 / ((mw_freqs_ghz - f_p)**2 + (self.gamma_fwhm / 2.0)**2)
        pl = 1.0 - contrast * (lor_m + lor_p)
        return pl
