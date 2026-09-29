from engine_45_extremophile_dna_radiation_repair import ExtremophileRadiationRepairEngine

def test_radiation_repair():
    engine = ExtremophileRadiationRepairEngine(num_blocks=4, block_size=32, seed=42)
    engine.inject_space_radiation_gamma(total_dose_krad=100.0)
    res = engine.reca_homologous_recombination_repair()
    assert res["memory_integrity_pct"] == 100.0
    assert res["residual_bit_errors"] == 0.0
