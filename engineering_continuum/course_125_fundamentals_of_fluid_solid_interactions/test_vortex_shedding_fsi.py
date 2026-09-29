from vortex_shedding_fsi import VortexSheddingFSI

def test_fsi_vibrations():
    f = VortexSheddingFSI.shedding_frequency(10.0, 0.5, 0.2)
    assert f == 4.0
    assert VortexSheddingFSI.check_lock_in(4.0, 10.0, 0.5) is True
