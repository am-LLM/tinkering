"""Course 066: MOSFET Gradual Channel Approximation (GCA) & Sub-Threshold Swing Engine"""
class MOSFETTransistor:
    def __init__(self, v_th=0.5, k_prime=200e-6, w_over_l=10.0, lambda_mod=0.02):
        self.v_th = v_th
        self.beta = k_prime * w_over_l
        self.lambda_mod = lambda_mod

    def drain_current(self, v_gs: float, v_ds: float) -> float:
        v_ov = v_gs - self.v_th
        if v_ov <= 0.0:
            return 0.0
        if v_ds < v_ov:
            return float(self.beta * (v_ov * v_ds - 0.5 * (v_ds ** 2)))
        else:
            return float(0.5 * self.beta * (v_ov ** 2) * (1.0 + self.lambda_mod * (v_ds - v_ov)))
