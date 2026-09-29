r"""
Cooperative Game-Theoretic Bargaining & Pie Theory Consensus Engine
===================================================================
Implements:
1. Barry Nalebuff "Split the Pie" Incremental Synergy Bargaining.
2. N-Player Characteristic Function Coalition Games $v(S)$.
3. Exact Combinatorial Shapley Value Payoff Calculator.
4. Generalized Nash Bargaining Solution with Asymmetric Bargaining Weights & Outside Options.
5. Dynamic Multi-Round Discounted Bargaining with Deadweight Friction Loss.
"""

from __future__ import annotations
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Callable, Set
import numpy as np


@dataclass
class PlayerProfile:
    player_id: str
    outside_option: float          # Disagreement / threat point d_i >= 0
    bargaining_power_weight: float = 1.0  # Asymmetric weight alpha_i > 0
    cost_contribution: float = 0.0 # Capital / compute resource invested


class PieTheoryBargaining:
    """
    Barry Nalebuff "Split the Pie" 2-Player & N-Player Bargaining Solver.
    The "Pie" is strictly the incremental surplus created by mutual cooperation:
    Pie = V_total - sum(Outside_Options).
    """

    @staticmethod
    def solve_two_party_pie(
        total_joint_value: float,
        player_a_outside: float,
        player_b_outside: float,
    ) -> Dict[str, float]:
        """
        Nalebuff 2-Party Equal Split of the Incremental Surplus:
        Payoff_A = a + Pie / 2
        Payoff_B = b + Pie / 2
        """
        sum_outside = player_a_outside + player_b_outside
        if total_joint_value < sum_outside:
            raise ValueError(
                f"Joint value ({total_joint_value}) is less than sum of outside options ({sum_outside}). "
                "Cooperation is irrational; players take outside options."
            )

        pie = total_joint_value - sum_outside
        half_pie = pie / 2.0

        payoff_a = player_a_outside + half_pie
        payoff_b = player_b_outside + half_pie

        return {
            "total_joint_value": total_joint_value,
            "net_cooperative_pie": pie,
            "player_a_payoff": payoff_a,
            "player_b_payoff": payoff_b,
            "player_a_gain": half_pie,
            "player_b_gain": half_pie,
            "is_pareto_optimal": True,
        }

    @staticmethod
    def compute_shapley_values(
        players: List[str],
        characteristic_fn: Callable[[FrozenSet[str]], float],
    ) -> Dict[str, float]:
        r"""
        Computes exact axiomatic Shapley values for an N-player cooperative game.
        phi_i(v) = sum_{S subseteq N \ {i}} [|S|!(|N| - |S| - 1)! / |N|!] * (v(S U {i}) - v(S))
        """
        n = len(players)
        if n == 0:
            return {}

        n_fact = math.factorial(n)
        player_set = set(players)
        shapley_values = {p: 0.0 for p in players}

        for i in players:
            other_players = list(player_set - {i})
            # Iterate through all subsets of other players
            for r in range(len(other_players) + 1):
                for subset in itertools.combinations(other_players, r):
                    s_size = len(subset)
                    weight = (math.factorial(s_size) * math.factorial(n - s_size - 1)) / n_fact
                    
                    s_frozen = frozenset(subset)
                    s_plus_i = frozenset(subset + (i,))

                    marginal_contrib = characteristic_fn(s_plus_i) - characteristic_fn(s_frozen)
                    shapley_values[i] += weight * marginal_contrib

        return shapley_values


class GeneralizedNashBargaining:
    """
    Generalized Nash Bargaining Solution (NBS) Optimizer.
    Solves: max prod_{i=1}^N (x_i - d_i)^{alpha_i} subject to sum(x_i) <= V and x_i >= d_i.
    """

    @staticmethod
    def solve_nash_equilibrium(
        total_value: float,
        players: List[PlayerProfile],
        tolerance: float = 1e-9,
        max_iterations: int = 1000,
    ) -> Dict[str, Any]:
        n = len(players)
        if n == 0:
            raise ValueError("Player list cannot be empty")

        d = np.array([p.outside_option for p in players], dtype=np.float64)
        alpha = np.array([p.bargaining_power_weight for p in players], dtype=np.float64)

        sum_d = np.sum(d)
        if total_value < sum_d:
            raise ValueError(f"Total value {total_value} cannot satisfy outside options sum {sum_d}")

        # Normalize weights
        alpha_normalized = alpha / np.sum(alpha)
        surplus = total_value - sum_d

        # Analytical closed-form for Generalized Nash Bargaining with linear frontier:
        # x_i = d_i + alpha_i_norm * surplus
        x_analytical = d + alpha_normalized * surplus

        # High-order constrained Newton-Raphson solver for logarithmic Nash frontier
        s = np.full(n, surplus / n, dtype=np.float64)
        for it in range(max_iterations):
            denom = np.sum((s ** 2) / alpha)
            delta_lambda = surplus / max(1e-12, denom)
            delta_s = s - delta_lambda * ((s ** 2) / alpha)
            s_next = np.maximum(s + delta_s, 1e-12)
            s_next = s_next * (surplus / np.sum(s_next))

            if np.max(np.abs(s_next - s)) < tolerance:
                s = s_next
                break
            s = s_next

        x = d + s

        # Check precision against analytical solution
        error = float(np.max(np.abs(x_analytical - x)))
        nash_product = float(np.prod((x_analytical - d) ** alpha_normalized))

        allocations = {players[i].player_id: float(x_analytical[i]) for i in range(n)}
        surplus_shares = {players[i].player_id: float(x_analytical[i] - d[i]) for i in range(n)}

        return {
            "total_value": total_value,
            "total_surplus_pie": float(surplus),
            "allocations": allocations,
            "surplus_shares": surplus_shares,
            "nash_product": nash_product,
            "convergence_error": error,
            "is_pareto_optimal": True,
        }


class DynamicMultiRoundBargaining:
    """
    Simulates multi-round sequential bargaining under discounting and deadweight friction loss.
    """

    def __init__(
        self,
        initial_pie: float,
        discount_factor_delta: float = 0.95,
        deadweight_loss_per_round: float = 2.0,
    ):
        if not (0.0 < discount_factor_delta <= 1.0):
            raise ValueError("Discount factor must be in (0.0, 1.0]")
        self.initial_pie = initial_pie
        self.delta = discount_factor_delta
        self.deadweight_loss = deadweight_loss_per_round

    def simulate_bargaining_trajectory(
        self,
        max_rounds: int = 10,
        agreement_round: int = 3,
        player_a_outside: float = 10.0,
        player_b_outside: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Simulates pie decay over rounds and calculates final payoffs upon agreement at round k.
        """
        rounds_history = []
        for r in range(max_rounds):
            effective_pie = max(0.0, self.initial_pie * (self.delta ** r) - r * self.deadweight_loss)
            rounds_history.append({
                "round": r,
                "available_pie": effective_pie,
                "discount_factor": self.delta ** r,
            })

        agree_pie = rounds_history[min(agreement_round, max_rounds - 1)]["available_pie"]
        payoffs = PieTheoryBargaining.solve_two_party_pie(
            total_joint_value=agree_pie + player_a_outside + player_b_outside,
            player_a_outside=player_a_outside,
            player_b_outside=player_b_outside,
        )

        return {
            "agreement_round": agreement_round,
            "final_pie": agree_pie,
            "payoffs": payoffs,
            "rounds_history": rounds_history,
        }
