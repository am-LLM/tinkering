from engine_07_photonic_dna_origami_router import PhotonicDNAOrigamiRouterEngine

def test_photonic_dna_router():
    engine = PhotonicDNAOrigamiRouterEngine(num_ports=4, seed=42)
    engine.hybridize({"SEQ_00": 1.0e-3, "SEQ_01": 5.0e-4})
    powers = engine.route_optical_signal(input_power_mw=10.0)
    assert "port_0_mW" in powers
    assert powers["total_input_power_mW"] == 10.0
