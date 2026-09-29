import numpy as np
from hrv_parasympathetic_analyzer import HRVParasympatheticAnalyzer

def test_hrv():
    rr = np.array([800, 850, 790, 860, 810])
    rmssd = HRVParasympatheticAnalyzer.calculate_rmssd(rr)
    assert rmssd > 0.0
