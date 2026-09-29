"""Course 075: 2D Convolution Forward Pass & Max-Pooling Feature Extractor"""
import numpy as np

class CNNFeatureExtractor:
    @staticmethod
    def conv2d(x: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
        h, w = x.shape
        kh, kw = kernel.shape
        out_h = (h - kh) // stride + 1
        out_w = (w - kw) // stride + 1
        out = np.zeros((out_h, out_w), dtype=np.float32)
        for i in range(out_h):
            for j in range(out_w):
                patch = x[i*stride:i*stride+kh, j*stride:j*stride+kw]
                out[i, j] = np.sum(patch * kernel)
        return out

    @staticmethod
    def max_pool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
        h, w = x.shape
        out_h = h // pool_size
        out_w = w // pool_size
        out = np.zeros((out_h, out_w), dtype=np.float32)
        for i in range(out_h):
            for j in range(out_w):
                out[i, j] = np.max(x[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size])
        return out
