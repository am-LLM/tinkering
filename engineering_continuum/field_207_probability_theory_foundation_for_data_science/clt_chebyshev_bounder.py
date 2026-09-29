"""Course 207: Chebyshev Inequality & Central Limit Theorem Probability Bounder"""
class CLTChebyshevBounder:
    @staticmethod
    def chebyshev_upper_bound(variance: float, epsilon: float) -> float:
        # P(|X - mu| >= epsilon) <= Var(X) / epsilon^2
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        return float(min(1.0, variance / (epsilon ** 2)))
