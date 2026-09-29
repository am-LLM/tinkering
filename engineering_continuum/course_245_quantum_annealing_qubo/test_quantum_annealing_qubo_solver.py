from quantum_annealing_qubo_solver import QuantumAnnealingQUBO
import numpy as np

def test_qubo_energy_minimization():
    # Simple 2-qubit antiferromagnetic interaction: Q = [[0, 2], [2, 0]]
    q = np.array([[2.0, -4.0], [-4.0, 2.0]])
    annealer = QuantumAnnealingQUBO(q)
    best_s, best_e = annealer.simulated_quantum_anneal(steps=150)
    assert len(best_s) == 2
    # The minimum is state [1, 1] giving 2 - 8 + 2 = -4
    assert best_e <= 0.0
