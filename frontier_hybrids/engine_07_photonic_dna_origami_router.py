"""Engine 07: Photonic Waveguide Mesh + DNA Origami Molecular Router."""
from dataclasses import dataclass
import numpy as np
from typing import Dict, List

@dataclass
class DNASwitchNode:
    node_id: int
    target_oligomer: str
    binding_gibbs_free_energy: float # kcal/mol
    hybridized_fraction: float = 0.0
    phase_shift_rad: float = 0.0

class PhotonicDNAOrigamiRouterEngine:
    def __init__(self, num_ports: int = 4, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_ports = num_ports
        self.nodes = [
            DNASwitchNode(node_id=i, target_oligomer=f"SEQ_0{i}",
                          binding_gibbs_free_energy=float(self.rng.uniform(-15.0, -8.0)))
            for i in range(num_ports)
        ]

    def hybridize(self, analyte_concentrations: Dict[str, float], temp_k: float = 310.15):
        """Compute hybridization equilibrium fraction via Gibbs isotherm."""
        R = 1.9872e-3 # kcal/(mol*K)
        for n in self.nodes:
            conc = analyte_concentrations.get(n.target_oligomer, 0.0)
            k_eq = np.exp(-n.binding_gibbs_free_energy / (R * temp_k))
            fraction = (k_eq * conc) / (1.0 + k_eq * conc + 1e-9)
            n.hybridized_fraction = float(np.clip(fraction, 0.0, 1.0))
            # Refractive index change induces optical phase shift in MZI waveguide
            n.phase_shift_rad = float(n.hybridized_fraction * np.pi)

    def route_optical_signal(self, input_power_mw: float = 10.0) -> Dict[str, float]:
        """Compute Mach-Zehnder optical routing crossbar matrix transmission."""
        output_powers = {}
        for n in self.nodes:
            # Transfer function: T_bar = cos^2(phi/2), T_cross = sin^2(phi/2)
            t_cross = np.sin(n.phase_shift_rad / 2.0) ** 2
            output_powers[f"port_{n.node_id}_mW"] = float(input_power_mw * t_cross)

        return {
            "total_input_power_mW": float(input_power_mw),
            **output_powers
        }
