import pytest
import numpy as np
from engineering_continuum.field_419_multichannel_eeg_welch_bandpower.eeg_welch_bandpower import MultichannelEEGSpectralExtractor

def test_field_419_initialization():
    extractor = MultichannelEEGSpectralExtractor(sample_rate=250.0, n_fft=128)
    assert extractor.sample_rate == 250.0
    assert "alpha" in extractor.bands

def test_field_419_dominant_alpha_detection():
    extractor = MultichannelEEGSpectralExtractor(sample_rate=250.0, n_fft=128)
    
    # Synthesize 10 Hz pure sine wave (alpha band) across 4 channels
    t = np.arange(500) / 250.0
    sig = np.sin(2 * np.pi * 10.0 * t)
    matrix = np.tile(sig, (4, 1))
    
    res = extractor.extract_band_powers(matrix)
    # Alpha should dominate relative powers
    assert res["relative_powers"]["alpha"] > 0.60
    assert bool(extractor.verify_properties()) is True
