"""Course 313: PLONK Permutation Argument & Grand Product Polynomial Copy Proofs"""
import numpy as np

class PLONKPermutationArgument:
    def __init__(self, nominal_scale: float = 1.0, channels: int = 4):
        self.scale = nominal_scale
        self.channels = channels
        self.history = []

    def compute_response(self, input_signal: np.ndarray = None) -> np.ndarray:
        x = np.ones(self.channels) if input_signal is None else np.array(input_signal, dtype=float)
        resp = np.tanh(x * self.scale) * self.scale
        self.history.append(resp)
        return resp

    def energy_metric(self) -> float:
        if not self.history:
            return 0.0
        return float(np.sum(np.square(self.history[-1])))

    def step_simulation(self, dt: float = 0.01) -> float:
        resp = self.compute_response()
        return float(np.mean(resp))
