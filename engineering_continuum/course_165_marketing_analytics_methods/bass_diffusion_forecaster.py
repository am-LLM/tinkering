"""Course 165: Bass Diffusion Model for New Product Adoption Forecasting"""
class BassDiffusionForecaster:
    def __init__(self, p_innovation: float = 0.03, q_imitation: float = 0.38, market_potential: float = 100000.0):
        self.p, self.q, self.m = p_innovation, q_imitation, market_potential

    def forecast_period(self, cumulative_adopters: float) -> float:
        # S(t) = p*(m - N) + (q/m)*N*(m - N)
        remaining = self.m - cumulative_adopters
        if remaining <= 0:
            return 0.0
        adopters = self.p * remaining + (self.q / self.m) * cumulative_adopters * remaining
        return float(adopters)
