"""Course 042: Rate Monotonic Scheduling (RMS) Feasibility & Dispatch Engine"""
import math
from typing import List, Dict

class RateMonotonicScheduler:
    def __init__(self, tasks: List[Dict]):
        # tasks format: [{"id": "T1", "c": execution_time, "t": period}]
        self.tasks = tasks

    def utilization(self) -> float:
        return sum(t["c"] / t["t"] for t in self.tasks)

    def liu_layland_bound(self) -> float:
        n = len(self.tasks)
        return n * (2 ** (1.0 / n) - 1.0)

    def is_schedulable_liu_layland(self) -> bool:
        return self.utilization() <= self.liu_layland_bound()

    def response_time_analysis(self) -> Dict[str, float]:
        # Sort tasks by period ascending (highest priority first)
        sorted_tasks = sorted(self.tasks, key=lambda x: x["t"])
        response_times = {}
        for i, task in enumerate(sorted_tasks):
            r = task["c"]
            while True:
                interference = sum(math.ceil(r / sorted_tasks[j]["t"]) * sorted_tasks[j]["c"] for j in range(i))
                r_next = task["c"] + interference
                if r_next == r:
                    response_times[task["id"]] = r
                    break
                if r_next > task["t"]:
                    response_times[task["id"]] = float("inf") # Deadline missed
                    break
                r = r_next
        return response_times
