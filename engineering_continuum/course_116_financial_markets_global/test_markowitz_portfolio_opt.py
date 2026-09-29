import numpy as np
from markowitz_portfolio_opt import MarkowitzPortfolioOptimizer

def test_markowitz():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = MarkowitzPortfolioOptimizer.min_variance_weights(cov)
    assert abs(np.sum(w) - 1.0) < 1e-4
