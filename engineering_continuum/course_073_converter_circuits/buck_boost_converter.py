"""Course 073: Non-Inverting Buck-Boost Converter State-Space Averaging Model"""
class BuckBoostConverter:
    def __init__(self, inductance_h=100e-6, capacitance_f=220e-6, resistance_ohm=10.0):
        self.l = inductance_h
        self.c = capacitance_f
        self.r = resistance_ohm

    def steady_state_gain(self, duty_cycle: float) -> float:
        if not (0.0 < duty_cycle < 1.0):
            raise ValueError("Duty cycle must be in (0, 1)")
        return float(duty_cycle / (1.0 - duty_cycle))

    def current_ripple(self, v_in: float, duty_cycle: float, f_sw: float) -> float:
        return float((v_in * duty_cycle) / (f_sw * self.l))

    def voltage_ripple(self, i_out: float, duty_cycle: float, f_sw: float) -> float:
        return float((i_out * duty_cycle) / (f_sw * self.c))
