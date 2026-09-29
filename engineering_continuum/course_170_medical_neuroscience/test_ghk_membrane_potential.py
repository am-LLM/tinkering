from ghk_membrane_potential import GHKVoltageModel

def test_ghk():
    vm = GHKVoltageModel.calculate_vm(1.0, 0.04, 0.45, 140, 5, 10, 145, 10, 110)
    assert -80.0 < vm < -50.0
