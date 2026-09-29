"""Course 114: Black-Scholes European Option Analytical Pricing & Greeks Engine"""
import math

class BlackScholesGreeks:
    @staticmethod
    def _cnd(d: float) -> float:
        return 0.5 * (1.0 + math.erf(d / math.sqrt(2.0)))

    @classmethod
    def call_price_and_greeks(cls, s: float, k: float, t: float, r: float, sigma: float) -> dict:
        d1 = (math.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        
        n_d1 = cls._cnd(d1)
        n_d2 = cls._cnd(d2)
        pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)
        
        price = s * n_d1 - k * math.exp(-r * t) * n_d2
        delta = n_d1
        gamma = pdf_d1 / (s * sigma * math.sqrt(t))
        vega = s * pdf_d1 * math.sqrt(t)
        theta = -(s * pdf_d1 * sigma) / (2.0 * math.sqrt(t)) - r * k * math.exp(-r * t) * n_d2
        
        return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}
