from density_matrix_lindblad_dissipation import LindbladOpenQuantumSystem

def test_lindblad_decoherence():
    system = LindbladOpenQuantumSystem()
    purities = [system.step() for _ in range(100)]
    assert purities[0] <= 1.0
    assert purities[-1] < purities[0], "Expected decoherence to decrease quantum state purity"
