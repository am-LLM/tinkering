"""Course 115: Cox-Ross-Rubinstein (CRR) Binomial Tree American Option Pricer"""
import math

class CRRBinomialPricer:
    @staticmethod
    def price_american_put(s0: float, k: float, t: float, r: float, sigma: float, steps: int = 50) -> float:
        dt = t / steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        p = (math.exp(r * dt) - d) / (u - d)
        discount = math.exp(-r * dt)
        
        # Terminal payoffs
        values = [max(0.0, k - s0 * (u ** j) * (d ** (steps - j))) for j in range(steps + 1)]
        
        # Backward induction
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                s_curr = s0 * (u ** j) * (d ** (i - j))
                hold = discount * (p * values[j + 1] + (1 - p) * values[j])
                exercise = max(0.0, k - s_curr)
                values[j] = max(hold, exercise)
                
        return float(values[0])
