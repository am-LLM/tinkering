"""Engine 34: CNT-FET Ballistic Transistor + Spiking In-Memory Convolution."""
from dataclasses import dataclass
import numpy as np
from typing import Dict

class CNTFETSpikingInMemConvEngine:
    def __init__(self, kernel_size: int = 3, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.k_size = kernel_size
        # Sobel edge kernel stored in CNT-FET crossbar conductance
        self.weights = np.array([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ], dtype=float)
        # CNT-FET on/off conductance ratio
        self.g_on = 1.0e-5 # 10 uS
        self.g_off = 1.0e-9 # 1 nS

    def convolve_spiking_patch(self, image_patch: np.ndarray, spike_threshold: float = 0.5) -> Dict[str, float]:
        """Perform analog dot-product in crossbar memory and generate output spike."""
        # Ballistic CNT-FET current I = G * V
        conductance_matrix = np.where(self.weights > 0, self.g_on * self.weights, self.g_off)
        analog_mac_current = float(np.sum(image_patch * conductance_matrix))
        
        # Integrate-and-fire spike generation
        membrane_potential = analog_mac_current * 1e6
        spike_fired = bool(membrane_potential >= spike_threshold)

        return {
            "mac_current_ua": float(analog_mac_current * 1e6),
            "membrane_potential": float(membrane_potential),
            "output_spike": 1.0 if spike_fired else 0.0
        }
