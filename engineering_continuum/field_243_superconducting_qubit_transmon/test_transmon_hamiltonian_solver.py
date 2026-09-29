from transmon_hamiltonian_solver import TransmonQubit
import numpy as np

def test_transmon_anharmonicity():
    qubit = TransmonQubit(ec=0.25, ej=15.0, ng=0.5)
    alpha = qubit.anharmonicity()
    assert alpha < 0  # Negative anharmonicity typical for transmon (-Ec)
    assert np.isclose(alpha, -0.25, atol=0.08)

def test_charge_dispersion_suppression():
    qubit = TransmonQubit(ec=0.25, ej=20.0)
    e0_0 = qubit.eigenenergies(k=1, ng_val=0.0)[0]
    e0_half = qubit.eigenenergies(k=1, ng_val=0.5)[0]
    # Charge dispersion should be exponentially suppressed for Ej/Ec >> 1
    assert abs(e0_0 - e0_half) < 1e-3
