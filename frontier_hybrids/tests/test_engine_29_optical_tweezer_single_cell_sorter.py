from engine_29_optical_tweezer_single_cell_sorter import OpticalTweezerCellSorterEngine

def test_optical_tweezer_sorter():
    engine = OpticalTweezerCellSorterEngine(seed=42)
    res = engine.step_sorting()
    assert res["sorted_targets"] == 4.0
    assert res["sorting_purity_pct"] == 100.0
