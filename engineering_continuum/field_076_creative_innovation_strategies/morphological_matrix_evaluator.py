"""Course 076: SCAMPER Morphological Analysis & Innovation Matrix Evaluator"""
from typing import List, Dict

class MorphologicalMatrix:
    def __init__(self):
        self.dimensions = {}

    def add_parameter(self, param_name: str, options: List[str]):
        self.dimensions[param_name] = options

    def total_combinations(self) -> int:
        if not self.dimensions:
            return 0
        prod = 1
        for opts in self.dimensions.values():
            prod *= len(opts)
        return prod

    def score_concept(self, concept: Dict[str, str], weights: Dict[str, float]) -> float:
        score = 0.0
        for param, opt in concept.items():
            if param in self.dimensions and opt in self.dimensions[param]:
                idx = self.dimensions[param].index(opt) + 1
                w = weights.get(param, 1.0)
                score += idx * w
        return float(score)
