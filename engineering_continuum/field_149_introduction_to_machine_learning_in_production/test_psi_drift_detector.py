import numpy as np
from psi_drift_detector import PSIDriftDetector

def test_psi():
    b = np.random.normal(0, 1, 1000)
    t_no_drift = np.random.normal(0, 1, 1000)
    t_drift = np.random.normal(2, 1, 1000)
    psi_low = PSIDriftDetector.calculate_psi(b, t_no_drift)
    psi_high = PSIDriftDetector.calculate_psi(b, t_drift)
    assert psi_low < 0.1
    assert psi_high > 0.25
