"""Course 191: Statistical Process Control (SPC) X-bar & R-Chart Control Limits"""
import numpy as np

class SPCControlCharts:
    @staticmethod
    def calculate_xbar_limits(subgroups: np.ndarray, a2_factor: float = 0.577) -> dict:
        # subgroups: shape (N_subgroups, n_sample_size)
        x_bars = np.mean(subgroups, axis=1)
        ranges = np.max(subgroups, axis=1) - np.min(subgroups, axis=1)
        
        x_double_bar = float(np.mean(x_bars))
        r_bar = float(np.mean(ranges))
        
        ucl = x_double_bar + a2_factor * r_bar
        lcl = x_double_bar - a2_factor * r_bar
        return {"x_double_bar": x_double_bar, "r_bar": r_bar, "ucl": ucl, "lcl": lcl}
