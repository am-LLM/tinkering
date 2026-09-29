"""Course 332: Clohessy-Wiltshire (CW) Relative Motion Orbital Rendezvous Engine"""
import numpy as np

class ClohessyWiltshirePropagator:
    def __init__(self, mean_motion_n: float = 0.0011): # ~90 min LEO orbit
        self.n = mean_motion_n

    def transition_matrix(self, dt: float) -> np.ndarray:
        nt = self.n * dt
        s = np.sin(nt)
        c = np.cos(nt)
        n = self.n
        
        phi = np.array([
            [4.0 - 3.0*c, 0.0, 0.0, s/n, (2.0/n)*(1.0-c), 0.0],
            [6.0*(s - nt), 1.0, 0.0, -(2.0/n)*(1.0-c), (4.0*s - 3.0*nt)/n, 0.0],
            [0.0, 0.0, c, 0.0, 0.0, s/n],
            [3.0*n*s, 0.0, 0.0, c, 2.0*s, 0.0],
            [-6.0*n*(1.0-c), 0.0, 0.0, -2.0*s, 4.0*c - 3.0, 0.0],
            [0.0, 0.0, -n*s, 0.0, 0.0, c]
        ])
        return phi

    def propagate(self, state_0: np.ndarray, dt: float) -> np.ndarray:
        phi = self.transition_matrix(dt)
        return phi @ state_0
