"""Course 079: Peak Current Mode Control with Slope Compensation"""
class PeakCurrentModeController:
    def __init__(self, compensation_slope: float = 0.5):
        self.se = compensation_slope

    def determine_switch_off(self, i_inductor: float, i_ref: float, time_in_period: float) -> bool:
        threshold = i_ref - self.se * time_in_period
        return i_inductor >= threshold

    def check_subharmonic_stability(self, m1: float, m2: float) -> bool:
        return self.se > (0.5 * m2 - m1) or m2 < m1
