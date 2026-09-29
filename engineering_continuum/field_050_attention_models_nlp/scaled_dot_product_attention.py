"""Course 050: Multi-Head Scaled Dot-Product Self-Attention Engine"""
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class ScaledDotProductAttention:
    @staticmethod
    def compute(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray = None):
        d_k = q.shape[-1]
        scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        weights = softmax(scores, axis=-1)
        output = weights @ v
        return output, weights
