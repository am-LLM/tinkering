from boost_converter_steady_state import BoostConverterModel

def test_boost_converter():
    boost = BoostConverterModel(v_in=12.0, r_load=20.0, f_sw=100e3)
    v_out = boost.v_out(0.5)
    assert round(v_out, 2) == 24.0
    delta_i = boost.inductor_current_ripple(0.5)
    assert delta_i > 0.0
