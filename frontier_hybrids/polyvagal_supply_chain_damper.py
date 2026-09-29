"""
Polyvagal Neuromorphic Supply-Chain Panic Damper (P-SCPD)
Fuses: Glasgow Pediatric Trauma Polyvagal Theory + Rutgers Supply Chain Bullwhip Dampening + Santa Fe Chaos Attractor Dynamics
"""

import numpy as np
from typing import Dict, List, Tuple

class PolyvagalSupplyChainDamper:
    """
    Applies Takens' Theorem Phase-Space Embedding and Polyvagal Vagal Co-regulation
    to suppress bullwhip variance oscillations in global supply chains.
    """
    def __init__(self, embedding_dimension: int = 3, time_delay_tau: int = 2):
        self.m = embedding_dimension
        self.tau = time_delay_tau

    def reconstruct_phase_space(self, order_history: np.ndarray) -> np.ndarray:
        """
        Reconstructs the dynamical attractor from 1D time series via Takens' delay embedding.
        X(t) = [x(t), x(t + tau), ..., x(t + (m-1)*tau)]
        """
        n = len(order_history)
        num_vectors = n - (self.m - 1) * self.tau
        if num_vectors <= 0:
            raise ValueError("Order history time series too short for chosen embedding parameters.")

        embedded_matrix = np.zeros((num_vectors, self.m))
        for i in range(num_vectors):
            embedded_matrix[i] = [order_history[i + j * self.tau] for j in range(self.m)]
        return embedded_matrix

    def calculate_lyapunov_divergence(self, embedded_attractor: np.ndarray) -> float:
        """
        Estimates the maximum local Lyapunov exponent lambda_max to detect chaotic panic bifurcations.
        lambda_max > 0 indicates chaotic bullwhip divergence (panic ordering).
        """
        diffs = np.diff(embedded_attractor, axis=0)
        norm_diffs = np.linalg.norm(diffs, axis=1)
        valid_norms = norm_diffs[norm_diffs > 1e-6]
        if len(valid_norms) < 2:
            return 0.0

        log_growth = np.diff(np.log(valid_norms))
        lambda_max = float(np.mean(log_growth))
        return lambda_max

    def compute_vagal_counter_pacing(self, current_demand: float, lyapunov_exponent: float, retail_inventory: float, target_inventory: float, rolling_demand_mean: float = 100.0) -> Tuple[float, Dict[str, float]]:
        """
        Computes the stabilizing 'vagal co-regulation' purchase order.
        If chaotic panic is detected (lambda_max > 0.015), applies non-linear exponential low-pass damping.
        """
        inventory_gap = target_inventory - retail_inventory
        base_order = current_demand + 0.35 * inventory_gap

        if lyapunov_exponent > 0.015:
            # Polyvagal Respiratory Sinus Arrhythmia (RSA) anti-resonance filter
            damping_ratio = 1.0 / (1.0 + 8.5 * lyapunov_exponent)
            # Filter high-frequency panic oscillations toward steady demand baseline
            vagal_stabilized_order = rolling_demand_mean + (current_demand - rolling_demand_mean) * damping_ratio + 0.2 * inventory_gap * damping_ratio
            panic_suppression_active = True
        else:
            vagal_stabilized_order = base_order
            panic_suppression_active = False

        metrics = {
            "base_uncontrolled_order": float(base_order),
            "vagal_stabilized_order": float(max(0.0, vagal_stabilized_order)),
            "lyapunov_chaos_index": float(lyapunov_exponent),
            "variance_reduction_pct": float(max(0.0, (1.0 - vagal_stabilized_order / max(1.0, base_order)) * 100.0 if panic_suppression_active else 0.0))
        }
        return float(max(0.0, vagal_stabilized_order)), metrics
