"""Course 127: 2x2 Normal Form Game Mixed Strategy Nash Equilibrium Solver"""
import numpy as np

class NashEquilibriumSolver:
    @staticmethod
    def solve_2x2_mixed(a: np.ndarray, b: np.ndarray) -> tuple:
        # Player 1 payoffs A, Player 2 payoffs B
        # p is P1 probability of action 1, q is P2 probability of action 1
        denom_q = (a[0,0] - a[0,1] - a[1,0] + a[1,1])
        q = (a[1,1] - a[0,1]) / denom_q if denom_q != 0 else 0.5
        
        denom_p = (b[0,0] - b[1,0] - b[0,1] + b[1,1])
        p = (b[1,1] - b[1,0]) / denom_p if denom_p != 0 else 0.5
        
        return float(np.clip(p, 0.0, 1.0)), float(np.clip(q, 0.0, 1.0))
