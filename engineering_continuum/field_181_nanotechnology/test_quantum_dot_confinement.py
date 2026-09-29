from quantum_dot_confinement import QuantumDotConfinement

def test_quantum_dot():
    ev = QuantumDotConfinement.ground_state_confinement_ev(2.5)
    assert ev > 0.0
