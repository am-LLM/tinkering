import numpy as np
from gaussian_mle_map import GaussianMLEMAP

def test_gaussian_mle():
    data = np.array([2.0, 4.0, 6.0])
    mu, s2 = GaussianMLEMAP.mle_estimates(data)
    assert mu == 4.0
    assert abs(s2 - 8.0/3.0) < 1e-4
