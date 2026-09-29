import numpy as np
from ardl_granger_causality import ARDLGrangerCausality

def test_granger():
    x = np.linspace(0, 10, 100)
    y = 2.0 * x + np.random.normal(0, 0.1, 100)
    f = ARDLGrangerCausality.granger_f_stat(y, x, lag=1)
    assert f >= 0.0
