"""Course 121: Multi-Touch Customer Attribution Modeling (Linear & Time-Decay)"""
class MultiTouchAttribution:
    @staticmethod
    def linear_attribution(touchpoints: list, conversion_value: float) -> dict:
        if not touchpoints:
            return {}
        credit_per_touch = conversion_value / len(touchpoints)
        att = {}
        for tp in touchpoints:
            att[tp] = att.get(tp, 0.0) + credit_per_touch
        return att

    @staticmethod
    def time_decay_attribution(touchpoints: list, half_life_days: float = 7.0, days_before: list = None, conversion_value: float = 100.0) -> dict:
        if days_before is None:
            days_before = list(range(len(touchpoints) - 1, -1, -1))
        weights = [2.0 ** (-d / half_life_days) for d in days_before]
        tot_w = sum(weights)
        att = {}
        for tp, w in zip(touchpoints, weights):
            att[tp] = att.get(tp, 0.0) + (w / tot_w) * conversion_value
        return att
