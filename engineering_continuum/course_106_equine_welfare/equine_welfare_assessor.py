"""Course 106: Equine Welfare Body Condition Score & Gait Asymmetry Evaluator"""
class EquineWelfareAssessor:
    @staticmethod
    def calculate_bcs(neck: float, ribs: float, pelvis: float) -> float:
        # Henneke 1-9 scale composite
        return round((neck + ribs + pelvis) / 3.0, 1)

    @staticmethod
    def gait_asymmetry_index(left_stance_time: float, right_stance_time: float) -> float:
        tot = left_stance_time + right_stance_time
        if tot == 0:
            return 0.0
        return abs(left_stance_time - right_stance_time) / tot
