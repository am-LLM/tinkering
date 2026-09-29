import numpy as np
from multirate_polyphase_dsp import PolyphaseDecimator

def test_decimator():
    taps = np.ones(5) / 5.0
    dec = PolyphaseDecimator(2, taps)
    sig = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    out = dec.process(sig)
    assert len(out) == 3
