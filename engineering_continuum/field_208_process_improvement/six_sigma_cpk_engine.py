"""Course 208: Six Sigma Process Capability Index (Cp, Cpk) & DPMO Calculator"""
import math

class SixSigmaCpkEngine:
    @staticmethod
    def calculate_cp_cpk(usl: float, lsl: float, mean: float, std: float) -> dict:
        cp = (usl - lsl) / (6.0 * std) if std > 0 else 0.0
        cpu = (usl - mean) / (3.0 * std) if std > 0 else 0.0
        cpl = (mean - lsl) / (3.0 * std) if std > 0 else 0.0
        cpk = min(cpu, cpl)
        return {"cp": float(cp), "cpk": float(cpk)}
