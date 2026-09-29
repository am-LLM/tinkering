import numpy as np
from ols_econometric_model import OLSEconometricModel

def test_ols_fit():
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    model = OLSEconometricModel()
    beta = model.fit(x, y)
    assert abs(beta[1] - 2.0) < 1e-4
    assert abs(model.r_squared - 1.0) < 1e-4
