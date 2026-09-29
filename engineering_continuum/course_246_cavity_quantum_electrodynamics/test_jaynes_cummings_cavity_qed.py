from jaynes_cummings_cavity_qed import JaynesCummingsCavityQED
import numpy as np

def test_vacuum_rabi_splitting():
    qed = JaynesCummingsCavityQED(omega_c=6.0, omega_a=6.0, g_coupling=0.2)
    e_plus, e_minus = qed.rabi_splitting_energies(n_photons=0)
    split = e_plus - e_minus
    assert np.isclose(split, 2.0 * 0.2)

def test_resonant_rabi_oscillations():
    qed = JaynesCummingsCavityQED(omega_c=5.0, omega_a=5.0, g_coupling=1.0)
    times = np.linspace(0, np.pi / 2.0, 50)
    prob = qed.excitation_probability_dynamics(times)
    assert np.isclose(prob[0], 1.0)
    assert np.isclose(prob[-1], 0.0, atol=1e-2)
