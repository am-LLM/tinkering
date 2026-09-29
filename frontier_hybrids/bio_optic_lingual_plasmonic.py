"""
Bio-Optic Lingual-Microbial Gasifier Synthesizer (L-MGS)
Fuses: Univ. of Cape Town (Severe Non-Verbal LIS Care) + Technion Nanosensors (Surface Plasmon Resonance) + Geneva Protocell Dynamics
"""

import numpy as np
from typing import Dict, Tuple

class SurfacePlasmonResonanceSolver:
    """
    Simulates Surface Plasmon Resonance (SPR) reflection dip shift (Kretschmann configuration)
    across gold-coated DVD grating prisms for non-contact exhaled breath volatile organic compound (VOC) detection.
    """
    def __init__(self, gold_dielectric_real: float = -12.5, gold_dielectric_imag: float = 1.2, prism_refractive_index: float = 1.517):
        self.eps_m_real = gold_dielectric_real
        self.eps_m_imag = gold_dielectric_imag
        self.n_p = prism_refractive_index

    def calculate_spr_resonance_angle(self, ambient_refractive_index: float) -> float:
        """
        Calculates the resonance coupling angle theta_spr:
        sin(theta_spr) = (1 / n_p) * sqrt( (eps_m_real * eps_d) / (eps_m_real + eps_d) )
        where eps_d = n_ambient^2
        """
        eps_d = ambient_refractive_index ** 2
        ratio = (self.eps_m_real * eps_d) / (self.eps_m_real + eps_d)
        if ratio < 0:
            return 45.0
        sin_theta = (1.0 / self.n_p) * np.sqrt(ratio)
        sin_theta = np.clip(sin_theta, 0.0, 0.999)
        theta_rad = np.arcsin(sin_theta)
        return float(np.degrees(theta_rad))

    def evaluate_breath_metabolic_distress(self, baseline_refractive_index: float, acetone_ppm: float, ammonia_ppm: float) -> Dict[str, any]:
        """
        Calculates the optical resonance shift delta_theta caused by exhaled biomarker gases.
        Elevated acetone (>1.8 ppm) -> Metabolic acidosis / fasting distress.
        Elevated ammonia (>0.9 ppm) -> Renal / gut inflammation overload.
        """
        # Gas index modulation sensitivity: delta_n = 1.2e-6 * acetone_ppm + 2.5e-6 * ammonia_ppm
        delta_n = (1.2e-6 * acetone_ppm) + (2.5e-6 * ammonia_ppm)
        current_n = baseline_refractive_index + delta_n

        theta_baseline = self.calculate_spr_resonance_angle(baseline_refractive_index)
        theta_current = self.calculate_spr_resonance_angle(current_n)
        shift_millidegrees = (theta_current - theta_baseline) * 1000.0

        metabolic_crisis = bool(acetone_ppm > 1.8 or ammonia_ppm > 0.9 or abs(shift_millidegrees) > 5.0)

        return {
            "spr_shift_millidegrees": float(shift_millidegrees),
            "acetone_ppm": float(acetone_ppm),
            "ammonia_ppm": float(ammonia_ppm),
            "metabolic_crisis_alert": metabolic_crisis,
            "distress_severity_score": float(np.clip(abs(shift_millidegrees) / 10.0, 0.0, 1.0))
        }
