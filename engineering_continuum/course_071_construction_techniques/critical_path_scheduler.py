"""Course 071: Critical Path Method (CPM) Construction Scheduling Engine"""
from typing import Dict, List, Set

class CriticalPathScheduler:
    def __init__(self):
        self.activities = {}
        self.precedences = {}

    def add_activity(self, name: str, duration: float, preds: List[str] = None):
        self.activities[name] = duration
        self.precedences[name] = preds or []

    def compute_schedule(self) -> dict:
        es = {act: 0.0 for act in self.activities}
        changed = True
        while changed:
            changed = False
            for act, dur in self.activities.items():
                preds = self.precedences[act]
                max_ef = max([es[p] + self.activities[p] for p in preds], default=0.0)
                if max_ef > es[act]:
                    es[act] = max_ef
                    changed = True
                    
        total_duration = max([es[a] + self.activities[a] for a in self.activities], default=0.0)
        lf = {a: total_duration for a in self.activities}
        changed = True
        while changed:
            changed = False
            for act, dur in self.activities.items():
                succs = [s for s, p_list in self.precedences.items() if act in p_list]
                if succs:
                    min_ls = min([lf[s] - self.activities[s] for s in succs])
                    if min_ls < lf[act]:
                        lf[act] = min_ls
                        changed = True

        critical_path = [a for a in self.activities if abs(es[a] - (lf[a] - self.activities[a])) < 1e-5]
        return {
            "project_duration": total_duration,
            "critical_path": critical_path,
            "early_start": es,
            "late_finish": lf
        }
