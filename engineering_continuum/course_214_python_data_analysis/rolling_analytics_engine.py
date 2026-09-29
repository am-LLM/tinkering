"""Course 214: Fast Rolling Window Statistics & Exponential Volatility Engine"""
import numpy as np

class RollingAnalyticsEngine:
    @staticmethod
    def rolling_sma_and_volatility(series: np.ndarray, window: int = 5) -> tuple:
        n = len(series)
        sma = np.array([np.mean(series[i:i+window]) for i in range(n - window + 1)])
        vol = np.array([np.std(series[i:i+window]) for i in range(n - window + 1)])
        return sma, vol
