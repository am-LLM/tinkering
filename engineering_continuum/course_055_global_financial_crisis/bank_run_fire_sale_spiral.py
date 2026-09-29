"""Course 055: Diamond-Dybvig Liquidity Panic & Fire-Sale Spillover Macroeconomic Engine"""
class BankLiquidityStressModel:
    def __init__(self, liquid_reserves=100.0, illiquid_assets=500.0, haircut_fire_sale=0.40):
        self.reserves = liquid_reserves
        self.assets = illiquid_assets
        self.haircut = haircut_fire_sale
        self.insolvent = False

    def process_withdrawal_shock(self, withdrawal_volume: float) -> dict:
        if withdrawal_volume <= self.reserves:
            self.reserves -= withdrawal_volume
        else:
            deficit = withdrawal_volume - self.reserves
            self.reserves = 0.0
            required_asset_liquidation = deficit / (1.0 - self.haircut)
            if required_asset_liquidation > self.assets + 1e-4:
                self.insolvent = True
                self.assets = 0.0
            else:
                self.assets = max(0.0, self.assets - required_asset_liquidation)
                
        return {
            "remaining_reserves": float(self.reserves),
            "remaining_assets": float(self.assets),
            "insolvent": self.insolvent
        }
