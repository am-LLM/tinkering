"""Course 003: Dopaminergic Micro-Reward Adaptive Interval Engine"""
import time

class ADHDIntervalScaffold:
    def __init__(self, base_work_min=15, micro_reward_points=10):
        self.work_interval = base_work_min
        self.points = 0
        self.reward_val = micro_reward_points

    def record_focus_streak(self, minutes_focused: float) -> int:
        if minutes_focused >= self.work_interval:
            earned = int(self.reward_val * (minutes_focused / self.work_interval))
            self.points += earned
            return earned
        return 0
