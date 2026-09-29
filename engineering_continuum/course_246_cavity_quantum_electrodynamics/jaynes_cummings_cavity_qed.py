"""Course 246: Jaynes-Cummings Cavity QED & Vacuum Rabi Splitting Engine"""
import numpy as np

class JaynesCummingsCavityQED:
    def __init__(self, omega_c: float = 5.0, omega_a: float = 5.0, g_coupling: float = 0.1, n_max: int = 10):
        self.omega_c = omega_c
        self.omega_a = omega_a
        self.g = g_coupling
        self.n_max = n_max

    def rabi_splitting_energies(self, n_photons: int) -> tuple[float, float]:
        delta = self.omega_a - self.omega_c
        rabi_freq = np.sqrt(delta**2 + 4.0 * (self.g**2) * (n_photons + 1))
        e_plus = (n_photons + 0.5) * self.omega_c + 0.5 * delta + 0.5 * rabi_freq
        e_minus = (n_photons + 0.5) * self.omega_c + 0.5 * delta - 0.5 * rabi_freq
        return float(e_plus), float(e_minus)

    def excitation_probability_dynamics(self, times: np.ndarray) -> np.ndarray:
        # Atomic excitation probability starting in |e, 0> resonant with cavity
        omega_r = 2.0 * self.g
        return np.cos(omega_r * times / 2.0)**2
