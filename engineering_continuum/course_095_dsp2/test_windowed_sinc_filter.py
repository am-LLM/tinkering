import numpy as np
from windowed_sinc_filter import WindowedSincFilter

def test_fir_filter():
    h = WindowedSincFilter.design_lowpass(0.2, 31)
    assert len(h) == 31
    assert abs(np.sum(h) - 1.0) < 1e-4
