import numpy as np
from radix2_fft_butterfly import radix2_fft
def test_radix2_fft_correctness():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    my_fft = radix2_fft(x)
    np_fft = np.fft.fft(x)
    assert np.allclose(my_fft, np_fft)
