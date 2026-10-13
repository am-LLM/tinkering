"""Course 429: Toroidal Closed-Loop Propeller Acoustic Tip-Vortex Attenuator"""
import numpy as np

class ToroidalPropellerAcousticNulling:
    def __init__(self, parameter: float = 1.0, channels: int = 4):
        self.param = parameter
        self.channels = channels
        self.state_buffer = np.zeros(channels, dtype=float)

    def process_telemetry(self, input_vector: np.ndarray = None) -> np.ndarray:
        vec = np.ones(self.channels) if input_vector is None else np.array(input_vector, dtype=float)
        resp = np.tanh(vec * self.param) * self.param
        self.state_buffer = resp
        return resp

    def compute_energy_metric(self) -> float:
        return float(np.sum(np.square(self.state_buffer)))

    def verify_properties(self) -> bool:
        res = self.process_telemetry()
        return bool(np.all(np.isfinite(res)) and len(res) == self.channels)
