"""
Quantum-Attribution Epigenetic Drone Beacon (QA-EDB)
Fuses: Meta Marketing Science (Multi-Touch Markov Attribution) + Quantum Optics 1 (SPDC g^(2)(0) Coherence) + UPenn Aerial Robotics
"""

import numpy as np
from typing import List, Dict, Tuple

class QuantumOpticalBeacon:
    """
    Simulates a Spontaneous Parametric Down-Conversion (SPDC) Entangled Photon Beacon.
    Measures second-order quantum coherence g^(2)(0) to create a physically unclonable optical signature (PUF).
    """
    def __init__(self, wavelength_nm: float = 810.0, pump_power_mw: float = 15.0):
        self.wavelength_nm = wavelength_nm
        self.pump_power_mw = pump_power_mw

    def calculate_second_order_coherence(self, dark_count_rate: float = 30.0, coincidence_rate: float = 1500.0) -> float:
        """
        Calculates the normalized intensity correlation function g^(2)(tau=0).
        For an authentic single-photon / entangled-pair source, g^(2)(0) < 0.5 (anti-bunching).
        Classical / thermal noise yields g^(2)(0) >= 1.0.
        """
        # Accidental-to-true coincidence ratio: R_acc = (N1 * N2 * tau_window) / N_coinc
        accidental_ratio = (dark_count_rate * dark_count_rate * 2e-9) / (coincidence_rate * 1e-6) if coincidence_rate > 0 else 1.0
        g2_0 = accidental_ratio
        return float(np.clip(g2_0, 0.005, 2.0))

    def verify_quantum_authenticity(self, g2_0: float) -> bool:
        """Single photon quantum authenticity barrier."""
        return g2_0 < 0.5


class MetaMarketingMarkovAttributionEngine:
    """
    Applies Multi-Touch Attribution Markov Transition Chains and Shapley Values
    to Electronic Warfare (EW) multi-emitter RF jamming source localization.
    """
    def __init__(self, emitter_ids: List[str]):
        self.emitter_ids = emitter_ids
        self.num_emitters = len(emitter_ids)
        self.states = ["START"] + emitter_ids + ["PACKET_DROP", "SUCCESS"]
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}

    def build_transition_matrix(self, journeys: List[List[str]]) -> np.ndarray:
        """
        Constructs a stochastic transition probability matrix from observed jamming exposure journeys.
        """
        n = len(self.states)
        trans_counts = np.zeros((n, n), dtype=np.float64)

        for journey in journeys:
            full_path = ["START"] + journey
            for i in range(len(full_path) - 1):
                u = self.state_to_idx[full_path[i]]
                v = self.state_to_idx[full_path[i + 1]]
                trans_counts[u, v] += 1.0

        # Row-normalize to get transition probabilities
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        trans_matrix = trans_counts / row_sums

        # Absorbing states
        drop_idx = self.state_to_idx["PACKET_DROP"]
        succ_idx = self.state_to_idx["SUCCESS"]
        trans_matrix[drop_idx, :] = 0.0
        trans_matrix[drop_idx, drop_idx] = 1.0
        trans_matrix[succ_idx, :] = 0.0
        trans_matrix[succ_idx, succ_idx] = 1.0

        return trans_matrix

    def calculate_removal_effects(self, trans_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculates the Removal Effect (Shapley-like causal attribution) for each jamming emitter.
        Removal Effect = 1.0 - (Probability of Packet Drop without Emitter / Base Probability of Packet Drop).
        """
        base_drop_prob = self._compute_drop_probability(trans_matrix)
        removal_effects = {}

        for emitter in self.emitter_ids:
            e_idx = self.state_to_idx[emitter]
            modified_matrix = trans_matrix.copy()
            # Remove emitter by zeroing incoming transitions and routing to SUCCESS
            modified_matrix[:, e_idx] = 0.0
            modified_matrix[e_idx, :] = 0.0
            modified_matrix[e_idx, self.state_to_idx["SUCCESS"]] = 1.0

            # Re-normalize rows
            row_sums = modified_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            modified_matrix = modified_matrix / row_sums

            drop_prob_without = self._compute_drop_probability(modified_matrix)
            effect = max(0.0, (base_drop_prob - drop_prob_without) / max(1e-5, base_drop_prob))
            removal_effects[emitter] = float(effect)

        # Normalize to attribution weights
        total_effect = sum(removal_effects.values())
        if total_effect > 0:
            for k in removal_effects:
                removal_effects[k] /= total_effect

        return removal_effects

    def _compute_drop_probability(self, P: np.ndarray) -> float:
        """Absorbing Markov chain fundamental matrix to find probability of absorption into PACKET_DROP."""
        # Split into transient (Q) and absorbing (R)
        transient_states = [s for s in self.states if s not in ["PACKET_DROP", "SUCCESS"]]
        t_indices = [self.state_to_idx[s] for s in transient_states]
        drop_idx = self.state_to_idx["PACKET_DROP"]

        Q = P[np.ix_(t_indices, t_indices)]
        R_drop = P[t_indices, drop_idx]

        I = np.eye(len(t_indices))
        try:
            N = np.linalg.inv(I - Q)
            B = N @ R_drop
            start_pos = transient_states.index("START")
            return float(np.clip(B[start_pos], 0.0, 1.0))
        except np.linalg.LinAlgError:
            return 0.5
