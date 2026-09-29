import numpy as np
from lstm_cell_engine import LSTMCellEngine

def test_lstm_forward():
    cell = LSTMCellEngine(input_dim=4, hidden_dim=8)
    x = np.ones(4)
    h = np.zeros(8)
    c = np.zeros(8)
    h_next, c_next = cell.forward(x, h, c)
    assert h_next.shape == (8,)
    assert c_next.shape == (8,)
