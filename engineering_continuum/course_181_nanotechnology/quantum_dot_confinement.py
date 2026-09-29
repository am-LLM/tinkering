"""Course 181: Quantum Dot Particle-in-a-Box Confinement Energy Calculator"""
class QuantumDotConfinement:
    H_BAR = 1.054571817e-34
    M_E = 9.1093837e-31

    @classmethod
    def ground_state_confinement_ev(cls, radius_nm: float, eff_mass_ratio: float = 0.13) -> float:
        r = radius_nm * 1e-9
        m_eff = eff_mass_ratio * cls.M_E
        energy_j = (cls.H_BAR ** 2 * 3.14159265 ** 2) / (2.0 * m_eff * (2.0 * r) ** 2)
        return float(energy_j / 1.602176634e-19)
