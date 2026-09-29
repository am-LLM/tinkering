import numpy as np
from pipeline_transformer_core import StandardScaleTransformer

def test_transformer():
    x = np.array([[1.0, 10.0], [3.0, 30.0]])
    tf = StandardScaleTransformer()
    xt = tf.fit_transform(x)
    assert np.allclose(np.mean(xt, axis=0), 0.0)
