from mosfet_gradual_channel_model import MOSFETTransistor

def test_mosfet_iv_characteristics():
    fet = MOSFETTransistor(v_th=0.5)
    i_cutoff = fet.drain_current(v_gs=0.3, v_ds=1.0)
    assert i_cutoff == 0.0
    i_lin = fet.drain_current(v_gs=1.5, v_ds=0.2)
    i_sat = fet.drain_current(v_gs=1.5, v_ds=2.0)
    assert i_sat > i_lin > 0.0
