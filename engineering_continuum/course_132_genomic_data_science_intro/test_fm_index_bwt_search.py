from fm_index_bwt_search import FMIndexBWTSearch

def test_bwt():
    bwt = FMIndexBWTSearch.bwt_transform("banana")
    assert len(bwt) == 7
    assert FMIndexBWTSearch.count_occurrences("banana", "an") == 2
