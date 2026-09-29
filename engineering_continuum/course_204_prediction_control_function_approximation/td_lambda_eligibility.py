"""Course 204: TD(lambda) Linear Value Function Approximation with Eligibility Traces"""
import numpy as np

class TDLambdaApproximator:
    def __init__(self, n_features: int, alpha: float = 0.05, gamma: float = 0.9, lambda_param: float = 0.8):
        self.weights = np.zeros(n_features)
        self.traces = np.zeros(n_features)
        self.alpha, self.gamma, self.lamb = alpha, gamma, lambda_param

    def update(self, phi: np.ndarray, reward: float, phi_next: np.ndarray):
        v_curr = np.dot(self.weights, phi)
        v_next = np.dot(self.weights, phi_next)
        td_error = reward + self.gamma * v_next - v_curr
        
        self.traces = self.gamma * self.lamb * self.traces + phi
        self.weights += self.alpha * td_error * self.traces
