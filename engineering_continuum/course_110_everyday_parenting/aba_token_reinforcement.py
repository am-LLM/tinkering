"""Course 110: Applied Behavior Analysis (ABA) Token Economy Reinforcement Tracker"""
class ABATokenReinforcement:
    def __init__(self, target_tokens: int = 10):
        self.target = target_tokens
        self.tokens = 0

    def reinforce(self, points: int = 1) -> bool:
        self.tokens += points
        return self.tokens >= self.target

    def redeem(self) -> bool:
        if self.tokens >= self.target:
            self.tokens -= self.target
            return True
        return False
