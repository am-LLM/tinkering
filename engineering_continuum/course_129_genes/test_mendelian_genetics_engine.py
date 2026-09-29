from mendelian_genetics_engine import MendelianGeneticsEngine

def test_genetics():
    hw = MendelianGeneticsEngine.hardy_weinberg(0.7)
    assert abs(hw["p2_homozygous_dom"] - 0.49) < 1e-4
    cross = MendelianGeneticsEngine.monohybrid_cross("Aa", "Aa")
    assert cross["AA"] == 0.25
