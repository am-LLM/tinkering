"""Course 229: Adverse Childhood Experience (ACE) & Resilience Protective Index"""
class ChildResilienceAssessor:
    @staticmethod
    def calculate_resilience_index(ace_score: int, protective_factors: int) -> dict:
        # Positive ratio: protective / (1 + ACE)
        ratio = protective_factors / (1.0 + ace_score)
        tier = "HIGH_RESILIENCE" if ratio >= 1.5 else "MODERATE" if ratio >= 0.8 else "VULNERABLE"
        return {"resilience_ratio": round(float(ratio), 2), "tier": tier}
