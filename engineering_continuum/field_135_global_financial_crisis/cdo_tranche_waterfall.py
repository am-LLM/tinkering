"""Course 135: Collateralized Debt Obligation (CDO) Tranche Loss Waterfall"""
class CDOTrancheWaterfall:
    def __init__(self, equity_pct: float = 0.05, mezz_pct: float = 0.15, senior_pct: float = 0.80):
        self.eq = equity_pct
        self.mezz = mezz_pct
        self.sen = senior_pct

    def evaluate_losses(self, total_loss_pct: float) -> dict:
        eq_loss = min(self.eq, total_loss_pct) / self.eq
        rem1 = max(0.0, total_loss_pct - self.eq)
        mezz_loss = min(self.mezz, rem1) / self.mezz if self.mezz > 0 else 0.0
        rem2 = max(0.0, rem1 - self.mezz)
        sen_loss = min(self.sen, rem2) / self.sen if self.sen > 0 else 0.0
        return {
            "equity_loss_pct": round(float(eq_loss), 4),
            "mezzanine_loss_pct": round(float(mezz_loss), 4),
            "senior_loss_pct": round(float(sen_loss), 4)
        }
