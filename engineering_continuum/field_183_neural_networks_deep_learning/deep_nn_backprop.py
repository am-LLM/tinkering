"""Course 183: Deep Neural Network Forward & Backpropagation Gradient Engine"""
import numpy as np

class DeepNNBackprop:
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    @classmethod
    def binary_cross_entropy_grad(cls, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        return a - y
