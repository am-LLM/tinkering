import numpy as np
from dna_methylation_ranker import DNAMethylationRanker

def test_methylation():
    b = np.array([0.1, 0.5, 0.9])
    m = DNAMethylationRanker.beta_to_m(b)
    assert m[1] == 0.0
