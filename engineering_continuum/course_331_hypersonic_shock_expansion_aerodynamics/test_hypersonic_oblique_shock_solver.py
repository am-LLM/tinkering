from hypersonic_oblique_shock_solver import HypersonicShockSolver
import numpy as np

def test_oblique_shock_pressure_jump():
    solver = HypersonicShockSolver(gamma=1.4)
    # Mach 6 flow, shock angle 30 deg -> Mn1 = 3.0
    p_ratio = solver.oblique_shock_pressure_ratio(mach_inf=6.0, shock_wave_angle_rad=np.radians(30.0))
    assert p_ratio > 1.0
    # For Mn1=3: 1 + (2.8/2.4) * (9 - 1) = 1 + 1.1667 * 8 = 10.333
    assert np.isclose(p_ratio, 10.3333, atol=0.1)

def test_newtonian_pressure_coeff():
    solver = HypersonicShockSolver()
    cp = solver.newtonian_impact_pressure_coefficient(np.radians(30.0))
    assert np.isclose(cp, 2.0 * (0.5**2))
