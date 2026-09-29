import numpy as np
from engine_05_metamaterial_organ_chip import AcousticOrganOnChipEngine

def test_acoustic_organ_chip():
    engine = AcousticOrganOnChipEngine(num_droplets=10, seed=42)
    focal = np.array([0.0, 0.0, 0.0])
    for _ in range(20):
        out = engine.step_microfluidics(focal_point=focal, dt=0.01)
    assert out["mean_displacement"] >= 0.0
