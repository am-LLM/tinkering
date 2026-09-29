"""Course 245: Adiabatic Quantum Annealing for MaxCut QUBO Solver"""
import numpy as np

class QuantumAnnealingQUBO:
    def __init__(self, qubo_matrix: np.ndarray, transverse_field_init: float = 5.0):
        self.q = np.array(qubo_matrix, dtype=float)
        self.n = len(self.q)
        self.gamma_0 = transverse_field_init

    def energy(self, spin_state: np.ndarray) -> float:
        # spin_state in {0, 1}^n
        return float(spin_state.T @ self.q @ spin_state)

    def simulated_quantum_anneal(self, steps: int = 100, beta_init: float = 0.1, beta_final: float = 10.0) -> tuple[np.ndarray, float]:
        state = np.random.choice([0, 1], size=self.n)
        best_state = state.copy()
        best_e = self.energy(best_state)

        for step in range(steps):
            s = step / float(steps)
            beta = beta_init * (1 - s) + beta_final * s
            gamma = self.gamma_0 * (1 - s)
            
            for i in range(self.n):
                candidate = state.copy()
                candidate[i] = 1 - candidate[i]
                de = self.energy(candidate) - self.energy(state) - gamma * (1.0 - 2.0 * s)
                if de < 0 or np.random.rand() < np.exp(-beta * de):
                    state = candidate
                    e_curr = self.energy(state)
                    if e_curr < best_e:
                        best_e = e_curr
                        best_state = state.copy()
        return best_state, best_e
