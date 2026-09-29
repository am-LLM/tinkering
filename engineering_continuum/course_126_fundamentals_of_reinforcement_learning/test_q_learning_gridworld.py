import numpy as np
from q_learning_gridworld import TabularQLearning

def test_q_learning():
    agent = TabularQLearning()
    agent.update(0, 1, 10.0, 1)
    assert agent.q_table[0, 1] > 0.0
