import numpy as np
from logistic_regression_logit import LogisticRegressionLogit

def test_logistic():
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    clf = LogisticRegressionLogit(lr=0.5, n_iter=200)
    clf.fit(x, y)
    assert clf.weights[0] > 0
    assert clf.odds_ratio(0) > 1.0
