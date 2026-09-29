"""Course 227: DQN Prioritized Experience Replay (PER) Memory Buffer"""
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 1000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def add(self, experience: tuple, priority: float = 1.0):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size: int = 4) -> list:
        probs = np.array(self.priorities) ** self.alpha
        probs /= np.sum(probs)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[i] for i in indices]
