"""Course 074: Type-II Digital Lead-Lag Voltage Compensator with Anti-Windup"""
class TypeIICompensator:
    def __init__(self, kp=1.2, ki=40.0, kd=0.01, v_ref=5.0, v_min=0.0, v_max=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.v_ref = v_ref
        self.v_min, self.v_max = v_min, v_max
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_duty(self, v_measured: float, dt: float = 1e-4) -> float:
        error = self.v_ref - v_measured
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        raw_duty = self.kp * error + self.ki * self.integral + self.kd * derivative
        clamped_duty = max(self.v_min, min(self.v_max, raw_duty))
        if raw_duty != clamped_duty:
            self.integral -= error * dt
        return float(clamped_duty)
