"""Course 107: Dual RC Thevenin Battery Equivalent Circuit Model"""
import math

class TheveninDualRC:
    def __init__(self, r0=0.01, r1=0.015, c1=1000.0, r2=0.02, c2=5000.0):
        self.r0, self.r1, self.c1, self.r2, self.c2 = r0, r1, c1, r2, c2
        self.v_rc1 = 0.0
        self.v_rc2 = 0.0

    def ocv_from_soc(self, soc: float) -> float:
        # Simple polynomial OCV(SOC)
        return 3.0 + 1.2 * soc

    def step(self, current_i: float, soc: float, dt: float = 1.0) -> float:
        # Positive current = discharge
        self.v_rc1 += (current_i / self.c1 - self.v_rc1 / (self.r1 * self.c1)) * dt
        self.v_rc2 += (current_i / self.c2 - self.v_rc2 / (self.r2 * self.c2)) * dt
        v_terminal = self.ocv_from_soc(soc) - current_i * self.r0 - self.v_rc1 - self.v_rc2
        return float(v_terminal)
