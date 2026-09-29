"""Course 046: Gray-Zone Escalation Threshold & Hybrid Warfare Game-Theoretic Engine"""
import numpy as np

class HybridConflictGame:
    def __init__(self, escalation_threshold=0.85, attribution_decay=0.92):
        self.threshold = escalation_threshold
        self.decay = attribution_decay
        self.attribution_score = 0.0
        self.instability_index = 0.1

    def execute_sub_threshold_action(self, action_vector: dict) -> dict:
        # action_vector: {"cyber_disruption": float, "info_ops": float, "infrastructure_probe": float}
        action_intensity = (
            0.35 * action_vector.get("cyber_disruption", 0.0) +
            0.25 * action_vector.get("info_ops", 0.0) +
            0.40 * action_vector.get("infrastructure_probe", 0.0)
        )
        # Update cumulative attribution with temporal decay
        self.attribution_score = self.attribution_score * self.decay + action_intensity * 0.3
        self.instability_index = float(np.clip(self.instability_index + action_intensity * 0.15, 0.0, 1.0))
        
        tripped_article_5 = self.attribution_score >= self.threshold
        return {
            "attribution": round(float(self.attribution_score), 4),
            "instability": round(float(self.instability_index), 4),
            "threshold_breached": tripped_article_5
        }
