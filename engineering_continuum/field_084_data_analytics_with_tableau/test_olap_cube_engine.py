from olap_cube_engine import OLAPDataCube

def test_olap():
    c = OLAPDataCube([{"k": "a", "v": 10}, {"k": "a", "v": 20}])
    assert c.aggregate(["k"], "v")[("a",)] == 30.0
