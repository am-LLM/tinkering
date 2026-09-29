from surface_code_syndrome_decoder import SurfaceCodeDistance3
import numpy as np

def test_surface_code_clean():
    sc = SurfaceCodeDistance3()
    x_err = np.zeros(9, dtype=int)
    z_err = np.zeros(9, dtype=int)
    z_syn, x_syn = sc.measure_syndrome(x_err, z_err)
    assert np.all(z_syn == 0)
    assert np.all(x_syn == 0)

def test_single_error_correction():
    sc = SurfaceCodeDistance3()
    x_err = np.zeros(9, dtype=int)
    x_err[0] = 1
    z_syn, _ = sc.measure_syndrome(x_err, np.zeros(9, dtype=int))
    correction = sc.decode_single_bit_flip(z_syn)
    residual = (x_err + correction) % 2
    assert np.all(residual == 0)
