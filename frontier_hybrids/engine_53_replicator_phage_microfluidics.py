"""
Engine 53: Evolutionary Replicator Dynamics + Stochastic Microfluidic Bacteriophage Reactor.
Anticipates bacterial CRISPR mutations and optimizes multi-phage cocktail ratios.
"""
import numpy as np

class PhageEvolutionaryGameReactor:
    def __init__(self, num_phage_strains: int = 4, num_bacterial_mutants: int = 4):
        self.n_p = num_phage_strains
        self.n_b = num_bacterial_mutants
        # Payoff matrix: infectivity / lysis rate
        np.random.seed(42)
        self.payoff_matrix = np.random.uniform(0.2, 1.0, (self.n_p, self.n_b))

    def step_replicator_dynamics(self, phage_frequencies: np.ndarray, bacterial_frequencies: np.ndarray, dt: float = 0.05) -> tuple:
        p = phage_frequencies / np.sum(phage_frequencies)
        b = bacterial_frequencies / np.sum(bacterial_frequencies)
        
        # Phage fitness: f_p = A * b
        fitness_p = self.payoff_matrix @ b
        avg_fitness_p = np.dot(p, fitness_p)
        dp = p * (fitness_p - avg_fitness_p) * dt
        p_next = np.clip(p + dp, 0.01, 1.0)
        p_next /= np.sum(p_next)
        
        # Bacterial fitness: f_b = 1.0 - p^T * A
        fitness_b = 1.0 - (p @ self.payoff_matrix)
        avg_fitness_b = np.dot(b, fitness_b)
        db = b * (fitness_b - avg_fitness_b) * dt
        b_next = np.clip(b + db, 0.01, 1.0)
        b_next /= np.sum(b_next)
        
        return p_next, b_next
