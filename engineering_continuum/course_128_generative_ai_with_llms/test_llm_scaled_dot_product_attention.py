import numpy as np
from llm_scaled_dot_product_attention import LLMScaledDotProductAttention

def test_attention():
    q = np.random.randn(2, 4, 8)
    k = np.random.randn(2, 4, 8)
    v = np.random.randn(2, 4, 8)
    out, weights = LLMScaledDotProductAttention.attention(q, k, v)
    assert out.shape == (2, 4, 8)
