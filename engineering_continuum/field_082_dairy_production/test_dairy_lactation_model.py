from dairy_lactation_model import DairyLactationModel

def test_dairy():
    m = DairyLactationModel()
    assert m.peak_day() == 5.0
