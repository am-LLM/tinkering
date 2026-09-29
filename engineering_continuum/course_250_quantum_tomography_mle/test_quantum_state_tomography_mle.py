from quantum_state_tomography_mle import QuantumStateTomographyMLE
import numpy as np

def test_tomography_pure_z_state():
    tomo = QuantumStateTomographyMLE()
    rho = tomo.reconstruct_density_matrix(exp_x=0.0, exp_y=0.0, exp_z=1.0)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.isclose(rho[0, 0], 1.0)
    assert np.isclose(rho[1, 1], 0.0)

def test_purity_constraint_projection():
    tomo = QuantumStateTomographyMLE()
    # Unphysical noisy measurements outside Bloch sphere
    rho = tomo.reconstruct_density_matrix(exp_x=0.8, exp_y=0.8, exp_z=0.8)
    evals = np.linalg.eigvalsh(rho)
    assert np.all(evals >= -1e-6)
