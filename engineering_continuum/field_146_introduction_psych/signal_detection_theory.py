"""Course 146: Signal Detection Theory (SDT) d-prime & Response Criterion Calculator"""
import math

class SignalDetectionTheory:
    @staticmethod
    def _inv_norm_cdf(p: float) -> float:
        # Rational approximation of inverse standard normal CDF
        p = max(1e-5, min(1.0 - 1e-5, p))
        # Winitzki approximation
        t = math.sqrt(-2.0 * math.log(min(p, 1.0 - p)))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z = t - ((c2*t + c1)*t + c0) / (((d3*t + d2)*t + d1)*t + 1.0)
        return -z if p < 0.5 else z

    @classmethod
    def calculate_sdt(cls, hit_rate: float, false_alarm_rate: float) -> dict:
        z_hr = cls._inv_norm_cdf(hit_rate)
        z_far = cls._inv_norm_cdf(false_alarm_rate)
        d_prime = z_hr - z_far
        criterion_c = -0.5 * (z_hr + z_far)
        return {"d_prime": float(d_prime), "criterion_c": float(criterion_c)}
