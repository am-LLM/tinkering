"""Course 082: Wood's Incomplete Gamma Lactation Curve Model"""
import math

class DairyLactationModel:
    def __init__(self, a: float = 15.0, b: float = 0.2, c: float = 0.04):
        self.a, self.b, self.c = a, b, c

    def daily_yield(self, day_in_milk: int) -> float:
        if day_in_milk <= 0:
            return 0.0
        return float(self.a * (day_in_milk ** self.b) * math.exp(-self.c * day_in_milk))

    def peak_day(self) -> float:
        return float(self.b / self.c)

    def cumulative_yield(self, days: int = 305) -> float:
        return float(sum(self.daily_yield(d) for d in range(1, days + 1)))
