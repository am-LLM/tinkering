import numpy as np
from policy_gradient_reinforce import PolicyGradientREINFORCE

def test_reinforce():
    w = np.zeros(2)
    s = [np.array([1.0, 0.0])]
    a = [1]
    g = [10.0]
    grad = PolicyGradientREINFORCE.compute_policy_gradient(s, a, g, w)
    assert grad[0] == 5.0
