"""Course 192: Autocatalytic Chemical Set Kinetics & Hypercycle ODE Integrator"""
class AutocatalyticNetworkSim:
    def __init__(self, k_catalytic: float = 0.5, d_decay: float = 0.05):
        self.k = k_catalytic
        self.d = d_decay

    def step_replicator(self, x: float, resource: float, dt: float = 0.1) -> float:
        # dx/dt = k * x * resource - d * x
        dx = self.k * x * resource - self.d * x
        return float(max(0.0, x + dx * dt))
