"""Engine 08: Anastomotic Mycelial Dynamics + HVDC Smart Grid Self-Healing."""
from dataclasses import dataclass
import numpy as np
from typing import Dict, List

@dataclass
class GridBus:
    bus_id: int
    generation_mw: float
    load_mw: float
    voltage_pu: float = 1.0
    tripped: bool = False

class MycelialSmartGridEngine:
    def __init__(self, num_buses: int = 6, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_buses = num_buses
        self.buses = [
            GridBus(bus_id=i,
                    generation_mw=float(self.rng.uniform(20.0, 100.0) if i % 2 == 0 else 0.0),
                    load_mw=float(self.rng.uniform(10.0, 50.0)))
            for i in range(num_buses)
        ]
        # Hyphal conductivity matrix (anastomotic network)
        self.hyphal_conductance = self.rng.uniform(0.5, 2.0, size=(num_buses, num_buses))
        self.hyphal_conductance = 0.5 * (self.hyphal_conductance + self.hyphal_conductance.T)
        np.fill_diagonal(self.hyphal_conductance, 0.0)

    def trip_bus(self, bus_id: int):
        if 0 <= bus_id < self.num_buses:
            self.buses[bus_id].tripped = True

    def step_mycelial_redistribution(self, dt: float = 0.1) -> Dict[str, float]:
        """Simulate hyphal resource flux (Pousielle nutrient flow) to reroute power."""
        active_buses = [b for b in self.buses if not b.tripped]
        total_gen = sum(b.generation_mw for b in active_buses)
        total_load = sum(b.load_mw for b in active_buses)

        # Anastomotic adaptation: strengthen links with high gradients
        for i in range(self.num_buses):
            for j in range(i + 1, self.num_buses):
                if not self.buses[i].tripped and not self.buses[j].tripped:
                    flow = (self.buses[i].generation_mw - self.buses[i].load_mw) -                            (self.buses[j].generation_mw - self.buses[j].load_mw)
                    # Hyphal thickening rule: dC/dt = alpha * |flow| - gamma * C
                    self.hyphal_conductance[i, j] += 0.05 * abs(flow) * dt - 0.01 * self.hyphal_conductance[i, j] * dt
                    self.hyphal_conductance[j, i] = self.hyphal_conductance[i, j]

        return {
            "total_generation_mw": float(total_gen),
            "total_load_mw": float(total_load),
            "grid_balance_deficit_mw": float(abs(total_gen - total_load)),
            "active_bus_ratio": float(len(active_buses) / self.num_buses)
        }
