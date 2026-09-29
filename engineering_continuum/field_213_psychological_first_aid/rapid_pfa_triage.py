"""Course 213: Johns Hopkins RAPID PFA Crisis Triage Priority Matrix"""
class RAPIDPFATriage:
    @staticmethod
    def triage_severity(cognitive_impairment: int, behavioral_arousal: int, functional_incapacity: int) -> dict:
        total_score = cognitive_impairment + behavioral_arousal + functional_incapacity
        tier = "HIGH_PRIORITY_INTERVENTION" if total_score >= 7 else "MONITOR_SUPPORT"
        return {"triage_score": total_score, "priority": tier}
