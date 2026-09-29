import numpy as np
from bollinger_ema_backtester import BollingerEMABacktester

def test_bollinger():
    p = np.linspace(100, 110, 30)
    sma, up, low = BollingerEMABacktester.compute_bollinger_bands(p, window=10)
    assert len(sma) == 21
    assert np.all(up >= low)
