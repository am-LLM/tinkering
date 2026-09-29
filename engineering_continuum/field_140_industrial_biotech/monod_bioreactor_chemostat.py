"""Course 140: Monod Microbial Growth Kinetics & Continuous Bioreactor Chemostat"""
class MonodBioreactorChemostat:
    def __init__(self, mu_max: float = 0.5, ks: float = 0.1, yield_coeff: float = 0.5):
        self.mu_max = mu_max
        self.ks = ks
        self.y = yield_coeff

    def specific_growth_rate(self, substrate_conc: float) -> float:
        if substrate_conc <= 0:
            return 0.0
        return float(self.mu_max * substrate_conc / (self.ks + substrate_conc))

    def steady_state_substrate(self, dilution_rate: float) -> float:
        # At steady state, mu = D
        if dilution_rate >= self.mu_max:
            raise ValueError("Washout occurs")
        return float((self.ks * dilution_rate) / (self.mu_max - dilution_rate))
