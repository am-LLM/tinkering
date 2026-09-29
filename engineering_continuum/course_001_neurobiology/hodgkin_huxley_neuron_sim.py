"""Course 001: Hodgkin-Huxley Membrane Potential Simulation"""
import numpy as np

class HodgkinHuxleyNeuron:
    def __init__(self, c_m=1.0, g_na=120.0, g_k=36.0, g_l=0.3, e_na=50.0, e_k=-77.0, e_l=-54.387):
        self.c_m, self.g_na, self.g_k, self.g_l = c_m, g_na, g_k, g_l
        self.e_na, self.e_k, self.e_l = e_na, e_k, e_l
        self.v, self.m, self.h, self.n = -65.0, 0.05, 0.60, 0.32

    def alpha_m(self, v): return 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))
    def beta_m(self, v): return 4.0 * np.exp(-(v + 65.0) / 18.0)
    def alpha_h(self, v): return 0.07 * np.exp(-(v + 65.0) / 20.0)
    def beta_h(self, v): return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
    def alpha_n(self, v): return 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))
    def beta_n(self, v): return 0.125 * np.exp(-(v + 65.0) / 80.0)

    def step(self, i_inj, dt=0.01):
        i_na = self.g_na * (self.m**3) * self.h * (self.v - self.e_na)
        i_k = self.g_k * (self.n**4) * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)
        dv = (i_inj - i_na - i_k - i_l) / self.c_m
        self.v += dv * dt
        self.m += (self.alpha_m(self.v) * (1 - self.m) - self.beta_m(self.v) * self.m) * dt
        self.h += (self.alpha_h(self.v) * (1 - self.h) - self.beta_h(self.v) * self.h) * dt
        self.n += (self.alpha_n(self.v) * (1 - self.n) - self.beta_n(self.v) * self.n) * dt
        return self.v
