"""Course 187: Long Short-Term Memory (LSTM) Cell Forward Gating Step"""
import numpy as np

class LSTMCellEngine:
    def __init__(self, input_dim: int, hidden_dim: int):
        self.h_dim = hidden_dim
        # Concatenated weights for [f, i, c_bar, o]
        self.w = np.random.randn(input_dim + hidden_dim, 4 * hidden_dim) * 0.1
        self.b = np.zeros(4 * hidden_dim)

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> tuple:
        concat = np.concatenate([x, h_prev], axis=-1)
        gates = np.dot(concat, self.w) + self.b
        
        f = 1.0 / (1.0 + np.exp(-gates[:self.h_dim]))
        i = 1.0 / (1.0 + np.exp(-gates[self.h_dim:2*self.h_dim]))
        c_bar = np.tanh(gates[2*self.h_dim:3*self.h_dim])
        o = 1.0 / (1.0 + np.exp(-gates[3*self.h_dim:]))
        
        c_next = f * c_prev + i * c_bar
        h_next = o * np.tanh(c_next)
        return h_next, c_next
