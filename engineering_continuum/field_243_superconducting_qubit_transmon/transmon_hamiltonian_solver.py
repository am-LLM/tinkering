"""Course 243: Superconducting Transmon Qubit Hamiltonian Solver"""
import numpy as np

class TransmonQubit:
    def __init__(self, ec: float = 0.25, ej: float = 15.0, ng: float = 0.5, n_cutoff: int = 10):
        self.ec = ec # Charging energy (GHz)
        self.ej = ej # Josephson energy (GHz)
        self.ng = ng # Offset charge
        self.n_cutoff = n_cutoff

    def build_hamiltonian(self, ng_val: float = None) -> np.ndarray:
        ng = self.ng if ng_val is None else ng_val
        n_dim = 2 * self.n_cutoff + 1
        n_diag = np.arange(-self.n_cutoff, self.n_cutoff + 1)
        h_charge = 4.0 * self.ec * np.diag((n_diag - ng)**2)
        h_josephson = -0.5 * self.ej * (np.diag(np.ones(n_dim - 1), 1) + np.diag(np.ones(n_dim - 1), -1))
        return h_charge + h_josephson

    def eigenenergies(self, k: int = 4, ng_val: float = None) -> np.ndarray:
        h = self.build_hamiltonian(ng_val=ng_val)
        evals = np.linalg.eigvalsh(h)
        return np.sort(evals)[:k]

    def anharmonicity(self) -> float:
        evals = self.eigenenergies(k=3)
        omega_01 = evals[1] - evals[0]
        omega_12 = evals[2] - evals[1]
        alpha = omega_12 - omega_01
        return float(alpha)
