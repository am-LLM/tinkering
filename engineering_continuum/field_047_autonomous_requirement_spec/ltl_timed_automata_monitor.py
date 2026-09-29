"""Course 047: Linear Temporal Logic (LTL) & Safety Invariant Runtime Verification Engine"""
from enum import Enum
from typing import List, Callable

class LTLOperator(Enum):
    ALWAYS = "G"   # Globally
    EVENTUALLY = "F" # In the future
    NEXT = "X"       # Next state

class LTLRuntimeMonitor:
    def __init__(self, invariants: List[Callable[[dict], bool]]):
        self.invariants = invariants
        self.violation_history = []
        self.step_count = 0

    def evaluate_state(self, state: dict) -> bool:
        self.step_count += 1
        for i, inv in enumerate(self.invariants):
            if not inv(state):
                self.violation_history.append({
                    "step": self.step_count,
                    "invariant_idx": i,
                    "failed_state": state
                })
                return False
        return True

    def is_safe(self) -> bool:
        return len(self.violation_history) == 0
