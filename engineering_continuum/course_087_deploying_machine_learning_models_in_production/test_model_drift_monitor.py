import numpy as np
from model_drift_monitor import DriftMonitor

def test_ks():
    m = DriftMonitor()
    a = np.zeros(100)
    b = np.ones(100)
    assert m.is_drift_detected(a, b, threshold=0.5) is True
