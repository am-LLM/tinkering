import numpy as np

class SpecializedEngine057:
    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def compute_transfer(self, x: float) -> float:
        return float(np.tanh(x * self.gain))

    def evaluate_energy_conservation(self, state_vector: list) -> float:
        return float(np.sum(np.square(state_vector)) * 0.5)
