from neural_ode_adjoint_engine import NeuralODEAdjointEngine
import numpy as np

def test_neuralodeadjointengine_transform():
    obj = NeuralODEAdjointEngine(parameter=1.2, length=5)
    out = obj.execute_transform(np.linspace(0, 1, 5))
    assert len(out) == 5
    assert obj.verify_properties()
    assert obj.get_norm() > 0.0

def test_neuralodeadjointengine_default():
    obj = NeuralODEAdjointEngine(parameter=0.5, length=3)
    out = obj.execute_transform()
    assert len(out) == 3
    assert np.allclose(out, np.cos(0.5) * 0.5)
