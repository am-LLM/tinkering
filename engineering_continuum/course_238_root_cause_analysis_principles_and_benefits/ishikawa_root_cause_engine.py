"""Course 238: Ishikawa Fishbone Diagram & 5-Whys Causal Hierarchy Engine"""
class IshikawaRootCauseEngine:
    CATEGORIES = ["Man", "Machine", "Material", "Method", "Measurement", "Environment"]

    def __init__(self, problem_statement: str):
        self.problem = problem_statement
        self.causes = {cat: [] for cat in self.CATEGORIES}

    def add_cause(self, category: str, cause: str):
        if category not in self.CATEGORIES:
            raise ValueError("Unknown Ishikawa 6M category")
        self.causes[category].append(cause)

    def total_causes_logged(self) -> int:
        return sum(len(c) for c in self.causes.values())
