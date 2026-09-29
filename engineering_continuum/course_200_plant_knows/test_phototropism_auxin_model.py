from phototropism_auxin_model import PhototropismAuxinModel

def test_auxin():
    curve = PhototropismAuxinModel.calculate_curvature_rate(70.0, 30.0, 0.01)
    assert curve == 0.4
