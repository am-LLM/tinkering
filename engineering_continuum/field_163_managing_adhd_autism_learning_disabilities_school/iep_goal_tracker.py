"""Course 163: Individualized Education Program (IEP) Accommodation & Mastery Engine"""
class IEPGoalTracker:
    def __init__(self, student_name: str):
        self.student = student_name
        self.goals = {}

    def add_goal(self, goal_id: str, description: str, target_pct: float):
        self.goals[goal_id] = {"desc": description, "target": target_pct, "scores": []}

    def record_progress(self, goal_id: str, score_pct: float):
        if goal_id in self.goals:
            self.goals[goal_id]["scores"].append(score_pct)

    def is_mastered(self, goal_id: str, window: int = 3) -> bool:
        scores = self.goals[goal_id]["scores"]
        if len(scores) < window:
            return False
        return all(s >= self.goals[goal_id]["target"] for s in scores[-window:])
