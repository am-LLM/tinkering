import numpy as np
from td_lambda_eligibility import TDLambdaApproximator

def test_td_lambda():
    td = TDLambdaApproximator(4)
    phi = np.array([1, 0, 0, 0])
    phi_next = np.array([0, 1, 0, 0])
    td.update(phi, 1.0, phi_next)
    assert td.weights[0] > 0.0
