from hill_muscle_tendon_dynamics import HillMuscleModel
import numpy as np

def test_hill_muscle_optimal_length():
    muscle = HillMuscleModel(f_max=2000.0, l_opt=0.1)
    fl_opt = muscle.force_length_active(0.1)
    fl_sub = muscle.force_length_active(0.05)
    assert np.isclose(fl_opt, 1.0)
    assert fl_sub < 1.0

def test_muscle_tendon_force_generation():
    muscle = HillMuscleModel(f_max=1000.0, l_slack=0.2, l_opt=0.1)
    force = muscle.compute_muscle_force(activation=0.8, l_ce=0.1, l_tendon=0.21)
    assert force > 0.0
    assert force <= 1000.0
