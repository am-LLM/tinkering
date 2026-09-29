"""Engine 02: Neuromorphic SNN Attractor + Jump-Diffusion Ruin Barrier."""
from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple

@dataclass
class SNNState:
    voltages: np.ndarray
    spikes: np.ndarray
    adaptation: np.ndarray
    threshold: float = 1.0
    decay: float = 0.9
    refractory: np.ndarray = None

class NeuromorphicJumpDiffusionEngine:
    def __init__(self, num_neurons: int = 32, initial_capital: float = 100.0, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_neurons = num_neurons
        self.capital = initial_capital
        self.ruin_barrier = 20.0
        self.is_ruined = False
        self.weights = self.rng.normal(0.05, 0.02, size=(num_neurons, num_neurons))
        np.fill_diagonal(self.weights, 0.0)
        self.snn = SNNState(
            voltages=np.zeros(num_neurons),
            spikes=np.zeros(num_neurons),
            adaptation=np.zeros(num_neurons),
            refractory=np.zeros(num_neurons, dtype=int)
        )

    def step(self, dt: float = 0.01, drift: float = 0.05, vol: float = 0.2, jump_lambda: float = 0.5) -> Dict[str, float]:
        if self.is_ruined:
            return {"capital": self.capital, "ruined": 1.0, "firing_rate": 0.0}

        # 1. SNN Dynamics: Attractor integration
        current_input = np.tanh(self.capital / 100.0) * 0.5 + self.rng.normal(0, 0.1, size=self.num_neurons)
        recurrent_input = self.weights @ self.snn.spikes
        total_input = current_input + recurrent_input - 0.2 * self.snn.adaptation

        self.snn.voltages = self.snn.voltages * self.snn.decay + total_input
        spikes = (self.snn.voltages >= self.snn.threshold).astype(float)
        self.snn.voltages[spikes > 0] = 0.0
        self.snn.adaptation = self.snn.adaptation * 0.95 + spikes * 0.1
        self.snn.spikes = spikes
        firing_rate = float(np.mean(spikes))

        # 2. Dynamic Hedging via Firing Rate: higher firing rate reduces jump sensitivity
        hedge_ratio = np.clip(firing_rate * 2.0, 0.1, 1.0)
        
        # 3. Merton Jump-Diffusion
        dW = self.rng.normal(0.0, np.sqrt(dt))
        n_jumps = self.rng.poisson(jump_lambda * dt)
        jump_size = 0.0
        if n_jumps > 0:
            jump_size = np.sum(self.rng.normal(-0.15, 0.1, size=n_jumps))

        d_capital = (drift * dt + vol * dW + jump_size * (1.0 - 0.5 * hedge_ratio)) * self.capital
        self.capital = max(0.0, self.capital + d_capital)

        if self.capital <= self.ruin_barrier:
            self.is_ruined = True

        return {
            "capital": float(self.capital),
            "ruined": 1.0 if self.is_ruined else 0.0,
            "firing_rate": firing_rate,
            "hedge_ratio": float(hedge_ratio)
        }
