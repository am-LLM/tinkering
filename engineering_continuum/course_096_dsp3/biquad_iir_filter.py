"""Course 096: Direct Form II Transposed Biquad IIR Filter Cascade"""
import numpy as np

class BiquadIIRFilter:
    def __init__(self, b: list, a: list):
        # H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
        a0 = a[0]
        self.b = [x / a0 for x in b]
        self.a = [x / a0 for x in a]
        self.s1 = 0.0
        self.s2 = 0.0

    def step(self, x: float) -> float:
        y = self.b[0] * x + self.s1
        self.s1 = self.b[1] * x - self.a[1] * y + self.s2
        self.s2 = self.b[2] * x - self.a[2] * y
        return float(y)
