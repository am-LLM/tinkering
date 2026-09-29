"""Engine 14: Solar Radiation Pressure + Geodesic Quaternion Orbital Navigation."""
from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple

@dataclass
class SailCraftState:
    position_au: np.ndarray # [x, y, z] in Astronomical Units
    velocity_kms: np.ndarray # [vx, vy, vz] in km/s
    quaternion: np.ndarray # [q0, q1, q2, q3]
    sail_area_m2: float = 1200.0
    craft_mass_kg: float = 100.0

class GeodesicSolarSailNavEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state = SailCraftState(
            position_au=np.array([1.0, 0.0, 0.0]),
            velocity_kms=np.array([0.0, 29.78, 0.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0])
        )
        self.p_sr_1au = 4.56e-6 # N/m^2 solar radiation pressure at 1 AU
        self.mu_sun = 1.327e11 # km^3/s^2

    def step_orbital_nav(self, sail_pitch_deg: float, dt_days: float = 1.0) -> Dict[str, float]:
        """Compute solar radiation pressure vector and integrate orbital geodesic."""
        dt_sec = dt_days * 86400.0
        r_au = float(np.linalg.norm(self.state.position_au))
        r_unit = self.state.position_au / r_au

        # Solar radiation pressure force: F = P * A * cos^2(alpha) * n_vector
        pitch_rad = np.radians(sail_pitch_deg)
        f_srp_mag = (2 * self.p_sr_1au / (r_au ** 2)) * self.state.sail_area_m2 * (np.cos(pitch_rad) ** 2)
        # Force direction in orbital plane
        normal_vec = np.array([np.cos(pitch_rad), np.sin(pitch_rad), 0.0])
        a_srp_kms2 = (f_srp_mag / self.state.craft_mass_kg) * 1e-3 * normal_vec

        # Solar gravity acceleration in km/s^2 (1 AU = 1.496e8 km)
        r_km = r_au * 1.496e8
        a_grav = - (self.mu_sun / (r_km ** 2)) * r_unit

        # Integrate
        self.state.velocity_kms += (a_srp_kms2 + a_grav) * dt_sec
        self.state.position_au += (self.state.velocity_kms * dt_sec) / 1.496e8

        return {
            "heliocentric_distance_au": float(np.linalg.norm(self.state.position_au)),
            "orbital_speed_kms": float(np.linalg.norm(self.state.velocity_kms)),
            "srp_acceleration_ms2": float(np.linalg.norm(a_srp_kms2) * 1e3)
        }
