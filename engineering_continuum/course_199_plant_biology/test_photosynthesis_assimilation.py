from photosynthesis_assimilation import PhotosynthesisAssimilation

def test_photosynthesis():
    a = PhotosynthesisAssimilation.calculate_net_assimilation(vc_max=100.0, ci=400.0)
    assert a > 0.0
