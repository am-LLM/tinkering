from r_dataframe_vector_ops import RDataFrameVectorOps

def test_r_ops():
    res = RDataFrameVectorOps.ifelse_vector([True, False, True], "YES", "NO")
    assert res == ["YES", "NO", "YES"]
    imp = RDataFrameVectorOps.impute_na_mean([10.0, None, 20.0])
    assert imp[1] == 15.0
