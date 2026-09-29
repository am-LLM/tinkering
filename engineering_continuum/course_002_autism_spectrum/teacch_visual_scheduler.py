"""Course 002: Structured TEACCH Visual Task Transition Engine"""
from enum import Enum
from typing import List, Dict

class TaskState(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class TEACCHScheduler:
    def __init__(self, visual_routine: List[str]):
        self.routine = [{ "task": t, "state": TaskState.PENDING } for t in visual_routine]
        self.current_idx = 0

    def start_next_task(self) -> Dict:
        if self.current_idx < len(self.routine):
            self.routine[self.current_idx]["state"] = TaskState.IN_PROGRESS
            return self.routine[self.current_idx]
        return None

    def complete_current_task(self) -> bool:
        if self.current_idx < len(self.routine):
            self.routine[self.current_idx]["state"] = TaskState.COMPLETED
            self.current_idx += 1
            return True
        return False
