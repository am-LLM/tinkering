from synchrophasor_dft_estimator import SynchrophasorDFTEstimator
import numpy as np

def test_synchrophasor_pure_sine():
    pmu = SynchrophasorDFTEstimator(nominal_freq=60.0, samples_per_cycle=48)
    t = np.arange(48) / (60.0 * 48)
    v_peak = 120.0 * np.sqrt(2.0)
    wave = v_peak * np.cos(2.0 * np.pi * 60.0 * t + 0.5)
    rms, phase, freq = pmu.estimate_phasor(wave)
    assert np.isclose(rms, 120.0, atol=1.0)
    assert np.isclose(freq, 60.0)
