import numpy as np
from deep_nn_backprop import DeepNNBackprop

def test_deep_nn():
    a = np.array([0.8, 0.2])
    y = np.array([1.0, 0.0])
    grad = DeepNNBackprop.binary_cross_entropy_grad(a, y)
    assert np.allclose(grad, [-0.2, 0.2])
