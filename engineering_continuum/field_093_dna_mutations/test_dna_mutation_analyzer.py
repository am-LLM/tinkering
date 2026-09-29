from dna_mutation_analyzer import DNAMutationAnalyzer

def test_mutation_analysis():
    orig = "AGCT"
    mut =  "GACT" # A->G (Ts), G->A (Ts)
    res = DNAMutationAnalyzer.calculate_ts_tv(orig, mut)
    assert res["transitions"] == 2
    assert res["transversions"] == 0
