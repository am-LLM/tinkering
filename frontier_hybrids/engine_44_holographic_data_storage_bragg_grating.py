"""Engine 44: Photopolymer Holographic Storage + Bragg Angle Multiplexing."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

class HolographicStorageBraggEngine:
    def __init__(self, page_dim: int = 16, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.dim = page_dim
        self.refractive_index_n1 = 1.5e-3
        self.medium_thickness_mm = 0.5
        self.wavelength_nm = 532.0 # Green laser
        self.stored_pages = {}

    def store_data_page(self, angle_deg: float, data_matrix: np.ndarray):
        """Record volumetric hologram at reference angle theta."""
        self.stored_pages[angle_deg] = np.array(data_matrix, dtype=float)

    def readout_page(self, probe_angle_deg: float) -> Dict[str, float]:
        """Kogelnik coupled wave theory: diffraction efficiency eta(theta)."""
        best_match_angle = None
        best_diffraction = 0.0
        
        for theta_stored in self.stored_pages.keys():
            delta_theta_rad = np.radians(probe_angle_deg - theta_stored)
            # Bragg selectivity parameter xi = 2 * pi * n * d * delta_theta / lambda
            xi = (2 * np.pi * 1.5 * (self.medium_thickness_mm * 1e-3) * delta_theta_rad) / (self.wavelength_nm * 1e-9)
            # Sinc squared angular selectivity
            sinc_factor = (np.sin(xi + 1e-9) / (xi + 1e-9)) ** 2
            eta = float(np.sin(np.pi * self.refractive_index_n1 * (self.medium_thickness_mm * 1e-3) / (self.wavelength_nm * 1e-9)) ** 2 * sinc_factor)
            
            if eta > best_diffraction:
                best_diffraction = eta
                best_match_angle = theta_stored

        crosstalk_noise = max(0.0, 1.0 - best_diffraction) if best_diffraction > 0.5 else 1.0
        return {
            "diffraction_efficiency": float(best_diffraction),
            "target_angle_matched_deg": float(best_match_angle) if best_match_angle is not None else -1.0,
            "inter_page_crosstalk": float(crosstalk_noise)
        }
