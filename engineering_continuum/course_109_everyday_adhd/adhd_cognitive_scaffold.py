"""Course 109: Executive Functioning Dynamic Micro-Task Load Balancer"""
class ADHDCognitiveScaffold:
    def __init__(self, energy_budget: int = 100):
        self.energy_budget = energy_budget
        self.tasks = []

    def add_task(self, name: str, cost: int, dopamine_reward: int):
        self.tasks.append({"name": name, "cost": cost, "reward": dopamine_reward})

    def get_optimal_sequence(self) -> list:
        # Sort by highest reward-to-cost ratio
        return sorted(self.tasks, key=lambda t: (t["reward"] / max(1, t["cost"])), reverse=True)
