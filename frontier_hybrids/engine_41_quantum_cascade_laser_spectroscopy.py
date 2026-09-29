"""Engine 41: Quantum Cascade Laser (QCL) + Mid-IR Trace Gas Detection."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

class QuantumCascadeSpectroscopyEngine:
    def __init__(self, emission_wavenumber_cm1: float = 2200.0, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.wavenumber = emission_wavenumber_cm1 # ~4.54 um (CO/CO2/N2O band)
        self.path_length_m = 10.0 # Multipass Herriott cell
        # Known absorption cross-section for CO at 2200 cm^-1
        self.sigma_cm2_molecule = 1.8e-18

    def detect_trace_gas(self, trace_gas_ppm: float, laser_power_mw: float = 15.0) -> Dict[str, float]:
        """Beer-Lambert law: I(nu) = I_0 * exp(-sigma * N * L)."""
        # Gas concentration to molecules/cm^3 (Loschmidt constant at STP ~ 2.68e19 cm^-3)
        n_density = (trace_gas_ppm * 1e-6) * 2.68e19
        path_cm = self.path_length_m * 100.0
        
        optical_depth = self.sigma_cm2_molecule * n_density * path_cm
        transmitted_power_mw = laser_power_mw * np.exp(-optical_depth)
        
        # Add photodetector thermal noise
        measured_power = float(transmitted_power_mw + self.rng.normal(0, 0.005))
        absorbance = float(-np.log(max(1e-6, measured_power / laser_power_mw)))
        
        # Recover estimated ppm
        estimated_ppm = float(absorbance / (self.sigma_cm2_molecule * 2.68e19 * 1e-6 * path_cm))

        return {
            "transmitted_power_mw": measured_power,
            "optical_absorbance": absorbance,
            "estimated_ppm": max(0.0, estimated_ppm),
            "detection_snr": float(absorbance / 0.005)
        }
