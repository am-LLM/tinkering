from elasticity_von_mises import VonMisesYieldCriterion

def test_von_mises():
    vm = VonMisesYieldCriterion.von_mises_stress(100.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert abs(vm - 100.0) < 1e-4
