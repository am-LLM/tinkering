from squeezed_vacuum_wigner import SqueezedStateEngine
import numpy as np

def test_squeezed_uncertainty_product():
    engine = SqueezedStateEngine(squeeze_param_r=1.2)
    var_x, var_p = engine.quadrature_variances()
    assert var_x < 0.5  # Sub-shot-noise in squeezed quadrature
    assert var_p > 0.5  # Anti-squeezed
    assert np.isclose(var_x * var_p, 0.25)  # Minimum uncertainty state

def test_wigner_normalization():
    engine = SqueezedStateEngine(squeeze_param_r=0.5)
    x = np.linspace(-3, 3, 100)
    p = np.linspace(-3, 3, 100)
    dx = x[1] - x[0]
    dp = p[1] - p[0]
    w = engine.wigner_function_2d(x, p)
    total_prob = np.sum(w) * dx * dp
    assert np.isclose(total_prob, 1.0, atol=1e-2)
