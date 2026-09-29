import numpy as np
from random_forest_bagging import BaggedStumpClassifier

def test_bagging():
    x = np.array([[1.0], [2.0], [8.0], [9.0]])
    y = np.array([0, 0, 1, 1])
    clf = BaggedStumpClassifier(10)
    clf.fit(x, y)
    pred = clf.predict(np.array([[1.5], [8.5]]))
    assert len(pred) == 2
