"""Course 159: Kelly Criterion Optimal Bet Sizing & Sortino Ratio Analyzer"""
import numpy as np

class KellySharpeOptimizer:
    @staticmethod
    def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
        # f* = p - (1-p)/b
        if win_loss_ratio <= 0:
            return 0.0
        f = win_prob - (1.0 - win_prob) / win_loss_ratio
        return float(max(0.0, f))

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
        excess = returns - risk_free_rate
        downside = returns[returns < target_return] - target_return
        downside_dev = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-6
        return float(np.mean(excess) / downside_dev)
