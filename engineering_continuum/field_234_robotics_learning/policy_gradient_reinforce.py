"""Course 234: Policy Gradient REINFORCE Log-Likelihood Gradient Estimator"""
import numpy as np

class PolicyGradientREINFORCE:
    @staticmethod
    def compute_policy_gradient(trajectories_states: list, actions: list, returns: list, weights: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(weights)
        for s, a, g in zip(trajectories_states, actions, returns):
            # Linear policy logits: pi(a|s) = softmax(w @ s)
            prob = 1.0 / (1.0 + np.exp(-np.dot(weights, s)))
            grad += (a - prob) * s * g
        return grad
