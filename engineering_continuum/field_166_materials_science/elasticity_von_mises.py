"""Course 166: 3D Stress Tensor & Von Mises Yield Stress Calculator"""
import math

class VonMisesYieldCriterion:
    @staticmethod
    def von_mises_stress(sxx: float, syy: float, szz: float, sxy: float, syz: float, szx: float) -> float:
        # sigma_v = sqrt(0.5 * [(sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2 + 6*(sxy^2 + syz^2 + szx^2)])
        term1 = (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
        term2 = 6.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
        return float(math.sqrt(0.5 * (term1 + term2)))
