from engine_20_hydrothermal_vent_chemosynthetic_alu import ChemosyntheticALUEngine

def test_chemosynthetic_alu():
    engine = ChemosyntheticALUEngine(seed=42)
    res = engine.step_metabolic_alu(op_a=1, op_b=1, control=0)
    assert res["carry_out"] == 1.0
    assert res["atp_pool"] > 0.0
