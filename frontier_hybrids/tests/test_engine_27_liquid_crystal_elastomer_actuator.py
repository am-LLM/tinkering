from engine_27_liquid_crystal_elastomer_actuator import LiquidCrystalElastomerEngine

def test_lce_crawler():
    engine = LiquidCrystalElastomerEngine(num_nodes=12, seed=42)
    for t in range(20):
        res = engine.step_peristaltic_crawl(thermal_stimulus_phase=t * 0.2, dt=0.05)
    assert "total_body_displacement_mm" in res
