"""Course 199: Farquhar C3 Photosynthesis Carbon Assimilation Model"""
class PhotosynthesisAssimilation:
    @staticmethod
    def calculate_net_assimilation(vc_max: float, ci: float, gamma_star: float = 40.0, kc: float = 300.0, ko: float = 200.0, o: float = 210.0, r_d: float = 1.0) -> float:
        # Rubisco-limited assimilation Ac = Vcmax * (Ci - Gamma*) / (Ci + Kc*(1 + O/Ko))
        k_app = kc * (1.0 + o / ko)
        ac = vc_max * (ci - gamma_star) / (ci + k_app)
        net_a = ac - r_d
        return float(net_a)
