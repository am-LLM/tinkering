"""Engine 05: Acoustic Metamaterial Levitation + Organ-on-Chip Drug Delivery."""
from dataclasses import dataclass
import numpy as np
from typing import Dict, List

@dataclass
class MicroDroplet:
    droplet_id: int
    position: np.ndarray # [x, y, z] in mm
    radius: float # in micrometers
    drug_payload: float # ng
    trapped: bool = False

class AcousticOrganOnChipEngine:
    def __init__(self, num_droplets: int = 15, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_droplets = num_droplets
        self.acoustic_freq = 2.0e6 # 2 MHz
        self.sound_speed = 1500.0 # m/s in fluid
        self.wavelength = self.sound_speed / self.acoustic_freq # 0.75 mm
        self.droplets = [MicroDroplet(droplet_id=i,
                                      position=self.rng.uniform(-2.0, 2.0, size=3),
                                      radius=float(self.rng.uniform(10.0, 50.0)),
                                      drug_payload=float(self.rng.uniform(1.0, 10.0)))
                         for i in range(num_droplets)]

    def gorkov_potential(self, pos: np.ndarray, focal_point: np.ndarray) -> float:
        """Gorkov acoustic radiation potential: U = 2*pi*a^3 * [ (p^2)/(3*rho*c^2) - (rho*v^2)/2 ]"""
        dist = np.linalg.norm(pos - focal_point)
        k = 2 * np.pi / self.wavelength
        potential = -np.exp(-0.5 * (dist / 0.5)**2) * np.cos(k * dist)
        return float(potential)

    def step_microfluidics(self, focal_point: np.ndarray, dt: float = 0.001) -> Dict[str, float]:
        trapped_count = 0
        total_delivered = 0.0

        for d in self.droplets:
            grad = (d.position - focal_point)
            dist = np.linalg.norm(grad)
            if dist < 0.2:
                d.trapped = True
                trapped_count += 1
                total_delivered += d.drug_payload
            else:
                d.trapped = False
                # Acoustic radiation force gradient descent towards trap
                f_rad = -grad / (dist + 1e-6) * 0.5
                fluid_drag = -0.1 * d.position
                d.position += (f_rad + fluid_drag) * dt

        return {
            "trapped_droplets": float(trapped_count),
            "total_drug_delivered_ng": float(total_delivered),
            "mean_displacement": float(np.mean([np.linalg.norm(d.position - focal_point) for d in self.droplets]))
        }
