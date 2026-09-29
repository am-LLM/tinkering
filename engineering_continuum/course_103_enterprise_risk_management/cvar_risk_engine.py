"""Course 103: Parametric & Historical Value-at-Risk (VaR) / CVaR Risk Engine"""
import numpy as np

class CVaRRiskEngine:
    @staticmethod
    def calculate_var_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> tuple:
        sorted_returns = np.sort(returns)
        alpha = 1.0 - confidence_level
        index = int(np.floor(alpha * len(sorted_returns)))
        var = -sorted_returns[index]
        cvar = -np.mean(sorted_returns[:index+1])
        return float(var), float(cvar)
