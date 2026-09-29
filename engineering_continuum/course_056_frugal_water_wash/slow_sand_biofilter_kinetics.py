"""Course 056: Slow Sand Biofilter Schmutzdecke Pathogen Degradation Kinetics"""
import numpy as np

class SlowSandFilter:
    def __init__(self, bed_depth_m=1.0, filtration_velocity_m_per_hr=0.15, k_bio_decay=0.25):
        self.depth = bed_depth_m
        self.velocity = filtration_velocity_m_per_hr
        self.k_bio = k_bio_decay

    def calculate_pathogen_removal_efficiency(self, schmutzdecke_age_days: int) -> float:
        biofilm_maturation = 1.0 - np.exp(-schmutzdecke_age_days / 14.0)
        effective_decay = self.k_bio * (1.0 + 2.0 * biofilm_maturation)
        retention_time_hr = self.depth / self.velocity
        log_reduction = effective_decay * retention_time_hr * 0.43429
        removal_pct = (1.0 - 10 ** (-log_reduction)) * 100.0
        return float(min(removal_pct, 99.99))
