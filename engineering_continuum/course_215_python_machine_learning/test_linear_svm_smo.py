import numpy as np
from linear_svm_smo import LinearSVMSMO

def test_svm_smo():
    x = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]])
    y = np.array([1, 1, -1, -1])
    svm = LinearSVMSMO(c=1.0)
    svm.fit_linear(x, y, max_passes=20)
    preds = svm.predict(np.array([[3.0, 3.0], [-3.0, -3.0]]))
    assert preds[0] == 1
    assert preds[1] == -1
