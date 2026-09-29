import numpy as np
from kfold_pr_evaluator import KFoldPREvaluator

def test_kfold():
    folds = KFoldPREvaluator.k_fold_split(100, k=5)
    assert len(folds) == 5
    assert len(folds[0][1]) == 20
