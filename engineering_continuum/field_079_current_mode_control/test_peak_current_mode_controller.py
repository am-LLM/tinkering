from peak_current_mode_controller import PeakCurrentModeController

def test_pcmc():
    c = PeakCurrentModeController(compensation_slope=1.0)
    assert c.check_subharmonic_stability(1.0, 2.0) is True
