import numpy as np
from cooley_tukey_fft import CooleyTukeyFFT

def test_fft_precision():
    x = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    my_fft = CooleyTukeyFFT.fft(x)
    np_fft = np.fft.fft(x)
    assert np.allclose(my_fft, np_fft)
