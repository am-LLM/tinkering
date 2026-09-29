"""Course 126: Tabular Q-Learning Bellman Optimality Engine"""
import numpy as np

class TabularQLearning:
    def __init__(self, num_states: int = 16, num_actions: int = 4, alpha: float = 0.1, gamma: float = 0.95):
        self.q_table = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma

    def update(self, s: int, a: int, r: float, s_next: int):
        best_next_a = np.max(self.q_table[s_next])
        td_error = r + self.gamma * best_next_a - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td_error

    def get_action(self, s: int) -> int:
        return int(np.argmax(self.q_table[s]))
