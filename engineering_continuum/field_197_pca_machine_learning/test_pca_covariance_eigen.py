import numpy as np
from pca_covariance_eigen import PCACovarianceEigen

def test_pca():
    x = np.random.randn(50, 4)
    proj, var = PCACovarianceEigen.compute_pca(x, 2)
    assert proj.shape == (50, 2)
    assert len(var) == 2
