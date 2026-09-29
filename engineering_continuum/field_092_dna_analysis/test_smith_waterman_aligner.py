from smith_waterman_aligner import SmithWatermanAligner

def test_smith_waterman():
    aligner = SmithWatermanAligner(match_score=3, mismatch_penalty=-1, gap_penalty=-2)
    score, pos = aligner.align("ACACACTA", "AGCACACA")
    assert score > 0
