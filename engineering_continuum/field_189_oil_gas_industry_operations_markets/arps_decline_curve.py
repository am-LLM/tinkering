"""Course 189: Arps Reservoir Production Decline Curve Analysis (Exponential & Hyperbolic)"""
import math

class ArpsDeclineCurve:
    @staticmethod
    def exponential_rate(qi: float, d: float, t: float) -> float:
        # q(t) = qi * exp(-d * t)
        return float(qi * math.exp(-d * t))

    @staticmethod
    def hyperbolic_rate(qi: float, di: float, b: float, t: float) -> float:
        # q(t) = qi / (1 + b * di * t)^(1/b)
        if b <= 0:
            return float(qi * math.exp(-di * t))
        return float(qi / ((1.0 + b * di * t) ** (1.0 / b)))
