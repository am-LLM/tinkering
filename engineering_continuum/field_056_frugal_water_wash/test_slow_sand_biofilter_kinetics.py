from slow_sand_biofilter_kinetics import SlowSandFilter

def test_slow_sand_filter():
    filter_unit = SlowSandFilter()
    eff_immature = filter_unit.calculate_pathogen_removal_efficiency(schmutzdecke_age_days=1)
    eff_mature = filter_unit.calculate_pathogen_removal_efficiency(schmutzdecke_age_days=21)
    assert eff_mature > eff_immature
    assert eff_mature > 85.0
