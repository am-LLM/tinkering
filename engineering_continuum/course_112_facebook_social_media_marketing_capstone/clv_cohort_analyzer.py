"""Course 112: Customer Lifetime Value (CLV) & Churn Survival Cohort Model"""
class CLVCohortAnalyzer:
    @staticmethod
    def calculate_clv(avg_order_value: float, purchase_freq_per_year: float, gross_margin: float, churn_rate_annual: float) -> float:
        if churn_rate_annual <= 0:
            return float('inf')
        annual_profit = avg_order_value * purchase_freq_per_year * gross_margin
        return float(annual_profit / churn_rate_annual)

    @staticmethod
    def cohort_retention_decay(initial_users: int, retention_rates: list) -> list:
        return [int(initial_users * r) for r in retention_rates]
