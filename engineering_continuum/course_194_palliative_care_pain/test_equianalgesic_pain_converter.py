from equianalgesic_pain_converter import EquianalgesicPainConverter

def test_mme_conversion():
    mme = EquianalgesicPainConverter.calculate_mme("oral_oxycodone", 20.0)
    assert mme == 30.0
