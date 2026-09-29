"""Course 043: Continuous Conduction Mode (CCM) Boost Converter Analytical Engine"""
class BoostConverterModel:
    def __init__(self, v_in=12.0, l_henry=220e-6, c_farad=470e-6, r_load=20.0, f_sw=100e3):
        self.v_in = v_in
        self.l = l_henry
        self.c = c_farad
        self.r = r_load
        self.f_sw = f_sw
        self.t_s = 1.0 / f_sw

    def voltage_conversion_ratio(self, duty_cycle: float) -> float:
        d = min(max(duty_cycle, 0.0), 0.95)
        return 1.0 / (1.0 - d)

    def v_out(self, duty_cycle: float) -> float:
        return self.v_in * self.voltage_conversion_ratio(duty_cycle)

    def inductor_current_ripple(self, duty_cycle: float) -> float:
        d = min(max(duty_cycle, 0.0), 0.95)
        # Delta i_L = (V_in * D * T_s) / (2 * L)
        return (self.v_in * d * self.t_s) / (2.0 * self.l)

    def output_voltage_ripple(self, duty_cycle: float) -> float:
        d = min(max(duty_cycle, 0.0), 0.95)
        v = self.v_out(duty_cycle)
        # Delta v_C = (V_out * D * T_s) / (2 * R * C)
        return (v * d * self.t_s) / (2.0 * self.r * self.c)
