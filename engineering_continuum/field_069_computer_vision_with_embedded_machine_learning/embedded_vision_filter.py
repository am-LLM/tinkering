"""Course 069: Fixed-Point Quantized Sobel Spatial Convolution Filter"""
import numpy as np

class EmbeddedSobelFilter:
    @staticmethod
    def convolve2d(image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        out = np.zeros((h - 2, w - 2), dtype=np.int32)
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int32)
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int32)
        for y in range(h - 2):
            for x in range(w - 2):
                patch = image[y:y+3, x:x+3]
                val_x = int(np.sum(patch * gx))
                val_y = int(np.sum(patch * gy))
                mag = int(np.sqrt(val_x**2 + val_y**2))
                out[y, x] = min(255, mag)
        return out
