"""Course 031: State-Space Averaged Buck Converter Model"""
import numpy as np

class BuckConverterStateSpace:
    def __init__(self, l_henry=100e-6, c_farad=220e-6, r_load=5.0, v_in=12.0):
        self.l, self.c, self.r, self.v_in = l_henry, c_farad, r_load, v_in
        self.i_l, self.v_c = 0.0, 0.0

    def step(self, duty_cycle: float, dt: float = 1e-6):
        d = np.clip(duty_cycle, 0.0, 1.0)
        # di_L/dt = (d * V_in - V_c) / L
        di_l = (d * self.v_in - self.v_c) / self.l
        # dv_C/dt = (i_L - V_c / R) / C
        dv_c = (self.i_l - self.v_c / self.r) / self.c
        self.i_l += di_l * dt
        self.v_c += dv_c * dt
        return float(self.v_c)
