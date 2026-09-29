import numpy as np
from mlp_adam_trainer import MLPAdamTrainer

def test_mlp():
    m = MLPAdamTrainer(2, 4, 1)
    out = m.forward(np.ones((1, 2)))
    assert out.shape == (1, 1)
