"""Course 239: First-Visit & Every-Visit Monte Carlo Policy Value Evaluator"""
class MCPolicyEvaluator:
    def __init__(self):
        self.returns = {}
        self.v = {}

    def update_first_visit(self, episode_states: list, episode_rewards: list, gamma: float = 0.9):
        # Calculate returns G_t
        g = 0.0
        visited = set()
        returns_list = []
        for r in reversed(episode_rewards):
            g = r + gamma * g
            returns_list.insert(0, g)
            
        for s, g_t in zip(episode_states, returns_list):
            if s not in visited:
                visited.add(s)
                self.returns.setdefault(s, []).append(g_t)
                self.v[s] = sum(self.returns[s]) / len(self.returns[s])
