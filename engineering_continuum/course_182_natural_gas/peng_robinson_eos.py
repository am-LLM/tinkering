"""Course 182: Peng-Robinson Cubic Equation of State Compressibility Factor Z"""
class PengRobinsonEOS:
    @staticmethod
    def calculate_reduced_params(t_k: float, p_bar: float, tc_k: float = 190.56, pc_bar: float = 45.99) -> tuple:
        t_r = t_k / tc_k
        p_r = p_bar / pc_bar
        return float(t_r), float(p_r)
