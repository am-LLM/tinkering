"""
Engine 63: Plant Stomatal Ion-Turgor Dynamics + Zero-Power Building Envelope.
"""
import numpy as np

class StomatalHydrogelBuildingSkin:
    def __init__(self, initial_aperture_mm: float = 2.0):
        self.aperture = initial_aperture_mm

    def step_turgor_ventilation(self, relative_humidity: float, co2_ppm: float) -> float:
        # Turgor pressure: Pi = R * T * (C_k + C_cl)
        osmotic_potential = (relative_humidity / 100.0) * 2.5 - (co2_ppm / 1000.0) * 0.8
        target_aperture = np.clip(self.aperture + 0.2 * osmotic_potential, 0.1, 10.0)
        self.aperture = float(target_aperture)
        return self.aperture
