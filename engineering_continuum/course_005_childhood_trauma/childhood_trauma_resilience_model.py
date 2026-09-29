import numpy as np

class TraumaResilienceModel:
    def __init__(self, ace_score: int):
        self.ace_score = ace_score
        self.vagal_tone = max(0.1, 1.0 - (ace_score * 0.08))

    def evaluate_allostatic_load(self, cortisol_baseline: float, chronic_stress_factor: float) -> float:
        return float(cortisol_baseline * (1.0 + self.ace_score * 0.15) * chronic_stress_factor / self.vagal_tone)

    def compute_resilience_index(self, supportive_adults_count: int, intervention_hours: float) -> float:
        protective_buffer = min(1.0, supportive_adults_count * 0.2 + (intervention_hours / 100.0) * 0.3)
        return float(np.clip(self.vagal_tone * (1.0 + protective_buffer), 0.0, 1.0))
