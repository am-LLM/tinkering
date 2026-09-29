import numpy as np
from sales_funnel_markov import SalesFunnelMarkov

def test_markov_funnel():
    # States: [Lead, Opp, Customer]
    t = np.array([
        [0.5, 0.3, 0.2],
        [0.0, 0.6, 0.4],
        [0.0, 0.0, 1.0]
    ])
    funnel = SalesFunnelMarkov(t, ["Lead", "Opp", "Customer"])
    out = funnel.simulate_absorption(np.array([1.0, 0.0, 0.0]), steps=20)
    assert out[2] > 0.5
