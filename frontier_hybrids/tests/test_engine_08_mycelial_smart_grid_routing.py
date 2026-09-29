from engine_08_mycelial_smart_grid_routing import MycelialSmartGridEngine

def test_mycelial_grid():
    engine = MycelialSmartGridEngine(num_buses=6, seed=42)
    engine.trip_bus(1)
    res = engine.step_mycelial_redistribution(dt=0.1)
    assert res["active_bus_ratio"] == 5.0 / 6.0
