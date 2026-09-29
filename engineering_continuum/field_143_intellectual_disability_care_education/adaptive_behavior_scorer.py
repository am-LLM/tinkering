"""Course 143: Adaptive Behavior Assessment System (ABAS) Composite Scorer"""
class AdaptiveBehaviorScorer:
    @staticmethod
    def composite_score(conceptual: float, social: float, practical: float) -> dict:
        gac = (conceptual + social + practical) / 3.0
        tier = "Average" if gac >= 90 else "Borderline" if gac >= 70 else "Mild/Moderate Support Required"
        return {"general_adaptive_composite": round(gac, 1), "support_tier": tier}
