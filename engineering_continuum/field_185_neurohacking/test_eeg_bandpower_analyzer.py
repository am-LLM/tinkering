import numpy as np
from eeg_bandpower_analyzer import EEGBandpowerAnalyzer

def test_eeg_bands():
    freqs = np.linspace(0, 50, 100)
    psd = np.ones(100)
    bp = EEGBandpowerAnalyzer.extract_bandpower(psd, freqs, 8.0, 12.0)
    assert abs(bp - 4.0) < 0.5
