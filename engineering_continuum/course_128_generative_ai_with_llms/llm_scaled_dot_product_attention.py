"""Course 128: Scaled Dot-Product Multi-Head Attention Mechanism with Causal Mask"""
import numpy as np

class LLMScaledDotProductAttention:
    @staticmethod
    def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray = None) -> tuple:
        d_k = q.shape[-1]
        scores = np.matmul(q, k.swapaxes(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        out = np.matmul(attn_weights, v)
        return out, attn_weights
