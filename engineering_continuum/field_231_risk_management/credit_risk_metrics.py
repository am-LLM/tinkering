"""Course 231: Credit Risk Expected Loss (EL = PD * LGD * EAD) & Capital Reserve"""
class CreditRiskMetrics:
    @staticmethod
    def calculate_expected_loss(pd: float, lgd: float, ead: float) -> float:
        # EL = Probability of Default * Loss Given Default * Exposure at Default
        return float(pd * lgd * ead)
