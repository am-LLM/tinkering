from solar_pv_single_diode import SolarPVSingleDiode

def test_pv_cell():
    i = SolarPVSingleDiode.cell_current(0.0) # Short circuit
    assert abs(i - 8.0) < 0.1
