"""Course 053: Statistical Process Control (SPC) Shewhart X-Bar & Jidoka Andon Alarm Engine"""
import numpy as np

class SPCProcessMonitor:
    def __init__(self, target_mu: float = 100.0, target_sigma: float = 2.0):
        self.mu = target_mu
        self.sigma = target_sigma
        self.ucl = target_mu + 3.0 * target_sigma
        self.lcl = target_mu - 3.0 * target_sigma
        self.history = []

    def evaluate_sample(self, sample_val: float) -> dict:
        self.history.append(sample_val)
        out_of_control = (sample_val > self.ucl) or (sample_val < self.lcl)
        run_alarm = False
        if len(self.history) >= 9:
            last_9 = self.history[-9:]
            all_above = all(x > self.mu for x in last_9)
            all_below = all(x < self.mu for x in last_9)
            run_alarm = all_above or all_below
            
        return {
            "sample": sample_val,
            "jidoka_andon_stop": bool(out_of_control or run_alarm),
            "ucl": self.ucl,
            "lcl": self.lcl
        }
