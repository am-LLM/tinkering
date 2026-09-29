"""Course 104: SEIR Compartmental Epidemic Model ODE Integrator"""
import numpy as np

class SEIREpidemicModel:
    def __init__(self, beta: float = 0.5, sigma: float = 0.2, gamma: float = 0.1, population: float = 10000.0):
        self.beta, self.sigma, self.gamma, self.n = beta, sigma, gamma, population

    def r0(self) -> float:
        return self.beta / self.gamma

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        s, e, i, r = state
        ds = -self.beta * s * i / self.n
        de = self.beta * s * i / self.n - self.sigma * e
        di = self.sigma * e - self.gamma * i
        dr = self.gamma * i
        return np.array([ds, de, di, dr])

    def step(self, state: np.ndarray, dt: float = 0.1) -> np.ndarray:
        return state + self.derivatives(state) * dt
