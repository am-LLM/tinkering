from ramsey_interferometry_phase_estimator import RamseyPhaseEstimator
import numpy as np

def test_ramsey_phase_response():
    estimator = RamseyPhaseEstimator(t_interrogation=0.001, detuning=500.0)
    p_zero = estimator.transition_probability(magnetic_shift=0.0)
    p_pi = estimator.transition_probability(magnetic_shift=np.pi / 0.001)
    assert 0.0 <= p_zero <= 1.0
    assert 0.0 <= p_pi <= 1.0
    assert abs(p_zero - p_pi) > 0.1

def test_sql_sensitivity():
    estimator = RamseyPhaseEstimator(t_interrogation=0.01)
    sens = estimator.estimate_phase_sensitivity(n_atoms=10000)
    assert sens > 0.0
    assert np.isclose(sens, 1.0 / (100.0 * 0.01))
