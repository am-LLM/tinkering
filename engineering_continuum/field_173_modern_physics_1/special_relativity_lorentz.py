"""Course 173: Special Relativity Lorentz 4-Vector Velocity Boost Engine"""
import math
import numpy as np

class LorentzTransform:
    C = 299792458.0

    @classmethod
    def gamma(cls, v: float) -> float:
        beta = v / cls.C
        return float(1.0 / math.sqrt(1.0 - beta ** 2))

    @classmethod
    def boost_x(cls, ct: float, x: float, v: float) -> tuple:
        g = cls.gamma(v)
        beta = v / cls.C
        ct_prime = g * (ct - beta * x)
        x_prime = g * (x - beta * ct)
        return float(ct_prime), float(x_prime)
