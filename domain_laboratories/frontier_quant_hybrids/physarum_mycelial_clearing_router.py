"""
Frontier Quant Hybrid Engine 2: Bio-Plasmodial Slime Mold (Physarum) & Mycelial Clearing Router (MYCO-ROUT)
Simulates adaptive protoplasmic tube conductivities and Hagen-Poiseuille fluid flux across cross-asset OTC/DEX graph nodes.
"""
import numpy as np
from typing import List, Dict, Tuple

class PhysarumClearingRouter:
    def __init__(self, num_nodes: int, gamma_decay: float = 0.05, mu_viscosity: float = 1.0):
        self.n = num_nodes
        self.gamma = gamma_decay
        self.mu = mu_viscosity
        # Conductivity matrix D_ij
        self.conductivity = np.ones((num_nodes, num_nodes)) * 0.1
        np.fill_diagonal(self.conductivity, 0.0)
        # Length / friction matrix L_ij
        self.length = np.ones((num_nodes, num_nodes))
        np.fill_diagonal(self.length, float("inf"))

    def set_edge(self, u: int, v: int, length: float, initial_conductance: float = 1.0):
        self.length[u, v] = length
        self.length[v, u] = length
        self.conductivity[u, v] = initial_conductance
        self.conductivity[v, u] = initial_conductance

    def solve_pressures(self, source: int, sink: int, input_flux: float = 1.0) -> np.ndarray:
        """Solves Poisson-like linear system for node pressures: sum_j D_ij / L_ij * (P_i - P_j) = S_i"""
        laplacian = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j and not np.isinf(self.length[i, j]):
                    k_ij = self.conductivity[i, j] / self.length[i, j]
                    laplacian[i, j] = -k_ij
                    laplacian[i, i] += k_ij
                    
        rhs = np.zeros(self.n)
        rhs[source] = input_flux
        rhs[sink] = -input_flux
        
        # Ground sink node to make system invertible
        laplacian_mod = laplacian.copy()
        laplacian_mod[sink, :] = 0.0
        laplacian_mod[sink, sink] = 1.0
        rhs_mod = rhs.copy()
        rhs_mod[sink] = 0.0
        
        pressures = np.linalg.solve(laplacian_mod, rhs_mod)
        return pressures

    def step_adaptation(self, source: int, sink: int, input_flux: float = 1.0, dt: float = 0.1) -> np.ndarray:
        """Evolves tube conductivities: dD_ij/dt = f(|Q_ij|) - gamma * D_ij"""
        p = self.solve_pressures(source, sink, input_flux)
        flux = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j and not np.isinf(self.length[i, j]):
                    # Hagen-Poiseuille flux Q_ij = (D_ij / L_ij) * (P_i - P_j)
                    q_ij = (self.conductivity[i, j] / (self.mu * self.length[i, j])) * (p[i] - p[j])
                    flux[i, j] = q_ij
                    # Conductance adaptation dD/dt
                    d_conductance = (abs(q_ij) - self.gamma * self.conductivity[i, j]) * dt
                    self.conductivity[i, j] = max(0.001, self.conductivity[i, j] + d_conductance)
        return flux
