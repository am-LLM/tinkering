from continuous_time_quantum_walk import ContinuousTimeQuantumWalk
import numpy as np

def test_quantum_walk_unitary_conservation():
    # 5-node ring graph
    adj = np.zeros((5, 5))
    for i in range(5):
        adj[i, (i+1)%5] = 1.0
        adj[(i+1)%5, i] = 1.0
    walk = ContinuousTimeQuantumWalk(adj)
    psi0 = np.zeros(5, dtype=complex)
    psi0[0] = 1.0
    psi_t = walk.evolve_state(psi0, t=2.5)
    prob = walk.probability_distribution(psi_t)
    assert np.isclose(np.sum(prob), 1.0)
    assert prob[0] < 1.0  # Distributed across nodes
