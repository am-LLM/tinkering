"""Course 068: Wilson-Cowan Excitatory-Inhibitory Neural Population Model"""
import numpy as np

class WilsonCowanModel:
    def __init__(self, c_ee=16.0, c_ie=12.0, c_ei=15.0, c_ii=3.0, tau_e=1.0, tau_i=2.0, theta_e=4.0, theta_i=3.7):
        self.c_ee, self.c_ie, self.c_ei, self.c_ii = c_ee, c_ie, c_ei, c_ii
        self.tau_e, self.tau_i = tau_e, tau_i
        self.theta_e, self.theta_i = theta_e, theta_i
        self.e = 0.1
        self.i = 0.1

    def sigmoid(self, x, theta, a=1.0):
        return 1.0 / (1.0 + np.exp(-a * (x - theta)))

    def step(self, p=1.2, q=0.0, dt=0.05):
        s_e = self.sigmoid(self.c_ee * self.e - self.c_ei * self.i + p, self.theta_e)
        s_i = self.sigmoid(self.c_ie * self.e - self.c_ii * self.i + q, self.theta_i)
        
        de = (-self.e + s_e) / self.tau_e
        di = (-self.i + s_i) / self.tau_i
        
        self.e += de * dt
        self.i += di * dt
        return float(self.e), float(self.i)
