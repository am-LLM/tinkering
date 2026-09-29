"""Course 148: Asymmetric INT8 Quantization & Affine Dequantization Engine"""
import numpy as np

class INT8Quantizer:
    @staticmethod
    def quantize(weights: np.ndarray) -> tuple:
        min_val = np.min(weights)
        max_val = np.max(weights)
        scale = (max_val - min_val) / 255.0 if max_val != min_val else 1.0
        zero_point = int(np.round(-min_val / scale)) - 128
        zero_point = max(-128, min(127, zero_point))
        
        q_weights = np.round(weights / scale) + zero_point
        q_weights = np.clip(q_weights, -128, 127).astype(np.int8)
        return q_weights, float(scale), zero_point

    @staticmethod
    def dequantize(q_weights: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        return (q_weights.astype(np.float32) - zero_point) * scale
