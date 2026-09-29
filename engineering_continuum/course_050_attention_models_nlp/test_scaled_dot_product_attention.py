import numpy as np
from scaled_dot_product_attention import ScaledDotProductAttention

def test_scaled_dot_product_attention():
    q = np.random.randn(2, 4, 16)
    k = np.random.randn(2, 4, 16)
    v = np.random.randn(2, 4, 16)
    out, weights = ScaledDotProductAttention.compute(q, k, v)
    assert out.shape == (2, 4, 16)
    assert np.allclose(np.sum(weights, axis=-1), 1.0)
