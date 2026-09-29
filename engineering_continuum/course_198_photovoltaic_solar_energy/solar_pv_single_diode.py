"""Course 198: Single-Diode 5-Parameter Photovoltaic Solar Cell Model"""
import math

class SolarPVSingleDiode:
    @staticmethod
    def cell_current(v: float, i_ph: float = 8.0, i_0: float = 1e-9, r_s: float = 0.01, r_sh: float = 1000.0, v_th: float = 0.026) -> float:
        # Approximate current using iterative fixed point
        i = i_ph
        for _ in range(5):
            exp_term = math.exp(min(40.0, (v + i * r_s) / v_th))
            i = i_ph - i_0 * (exp_term - 1.0) - (v + i * r_s) / r_sh
        return float(i)
