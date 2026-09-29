from engine_13_thermoacoustic_stirling_cryocooler import ThermoacousticCryocoolerEngine

def test_cryocooler():
    engine = ThermoacousticCryocoolerEngine(seed=42)
    res = engine.step_cryocooling(heat_load_watts=2.0, dt=0.1)
    assert 4.0 <= res["cold_temp_k"] <= 300.0
    assert res["acoustic_work_watts"] > 0.0
