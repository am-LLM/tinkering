from synchronous_boost_model import SynchronousBoostModel

def test_boost_model():
    res = SynchronousBoostModel.small_signal_gvd(v_out=24.0, duty=0.5, r_load=10.0, l_h=100e-6, c_f=100e-6)
    assert res["dc_gain"] == 48.0
    assert res["rhpz_rad_s"] > 0
