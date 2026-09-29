"""Course 168: Campaign ROAS, CAC & Blended Marketing Efficiency Ratio Engine"""
class ROASCACOptimizer:
    @staticmethod
    def evaluate_campaign(ad_spend: float, revenue_generated: float, new_customers: int) -> dict:
        roas = revenue_generated / ad_spend if ad_spend > 0 else 0.0
        cac = ad_spend / new_customers if new_customers > 0 else float('inf')
        return {"roas": float(roas), "cac": float(cac)}
