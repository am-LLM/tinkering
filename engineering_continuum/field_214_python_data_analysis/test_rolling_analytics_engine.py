import numpy as np
from rolling_analytics_engine import RollingAnalyticsEngine

def test_rolling():
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sma, vol = RollingAnalyticsEngine.rolling_sma_and_volatility(s, window=3)
    assert len(sma) == 4
    assert sma[0] == 2.0
