"""Course 080: Multi-Domain NIST CSF Security Risk & Control Matrix Engine"""
from typing import Dict, List

class NISTRiskEngine:
    DOMAINS = ["Identify", "Protect", "Detect", "Respond", "Recover"]

    def __init__(self):
        self.controls = {d: [] for d in self.DOMAINS}

    def add_control(self, domain: str, name: str, weight: float, score: float):
        if domain not in self.DOMAINS:
            raise ValueError("Unknown NIST domain")
        self.controls[domain].append({"name": name, "weight": weight, "score": score})

    def domain_maturity(self, domain: str) -> float:
        ctrls = self.controls[domain]
        if not ctrls:
            return 0.0
        tot_w = sum(c["weight"] for c in ctrls)
        tot_score = sum(c["weight"] * c["score"] for c in ctrls)
        return float(tot_score / tot_w if tot_w > 0 else 0.0)

    def overall_posture(self) -> float:
        scores = [self.domain_maturity(d) for d in self.DOMAINS if self.controls[d]]
        return float(sum(scores) / len(scores) if scores else 0.0)
