"""Course 200: Cholodny-Went Lateral Auxin Redistribution & Phototropism Model"""
class PhototropismAuxinModel:
    @staticmethod
    def calculate_curvature_rate(auxin_shaded: float, auxin_lit: float, sensitivity_coeff: float = 0.02) -> float:
        # Differential elongation rate proportional to delta auxin
        delta_auxin = auxin_shaded - auxin_lit
        return float(sensitivity_coeff * delta_auxin)
