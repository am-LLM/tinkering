import numpy as np
from statistical_anomaly_detector import StatisticalAnomalyDetector

def test_stats_anomaly():
    arr = np.array([1, 1, 1, 1, 100])
    res = StatisticalAnomalyDetector.iqr_anomalies(arr)
    assert bool(res[-1]) is True
