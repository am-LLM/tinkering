"""Course 145: Bollinger Band Mean-Reversion & EMA Crossover Strategy Backtester"""
import numpy as np

class BollingerEMABacktester:
    @staticmethod
    def compute_bollinger_bands(prices: np.ndarray, window: int = 20, num_std: float = 2.0) -> tuple:
        sma = np.convolve(prices, np.ones(window)/window, mode='valid')
        rolling_std = np.array([np.std(prices[i:i+window]) for i in range(len(prices) - window + 1)])
        upper = sma + num_std * rolling_std
        lower = sma - num_std * rolling_std
        return sma, upper, lower
