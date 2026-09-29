from engine_16_superconducting_flux_qubit_fpga import SuperconductingFluxQDIAlgebraEngine

def test_superconducting_fpga():
    engine = SuperconductingFluxQDIAlgebraEngine(num_junctions=8, seed=42)
    engine.inject_fluxon_pulse(2)
    res = engine.step_soliton_transport(bias_current_ratio=0.8, dt=0.01)
    assert res["active_fluxons"] >= 1
