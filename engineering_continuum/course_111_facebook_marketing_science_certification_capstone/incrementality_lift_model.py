"""Course 111: Marketing Incrementality Lift & Conversion Rate Test Engine"""
import math

class IncrementalityLiftModel:
    @staticmethod
    def calculate_lift(test_conversions: int, test_size: int, control_conversions: int, control_size: int) -> dict:
        p_test = test_conversions / max(1, test_size)
        p_ctrl = control_conversions / max(1, control_size)
        abs_lift = p_test - p_ctrl
        rel_lift = (abs_lift / p_ctrl) if p_ctrl > 0 else 0.0
        
        # Two-proportion z-statistic
        p_pool = (test_conversions + control_conversions) / (test_size + control_size)
        se = math.sqrt(p_pool * (1 - p_pool) * (1/test_size + 1/control_size)) if p_pool * (1-p_pool) > 0 else 1.0
        z_stat = abs_lift / se if se > 0 else 0.0
        return {"abs_lift": abs_lift, "relative_lift": rel_lift, "z_statistic": z_stat}
