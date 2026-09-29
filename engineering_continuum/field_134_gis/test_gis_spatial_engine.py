from gis_spatial_engine import GISSpatialEngine

def test_gis_engine():
    d = GISSpatialEngine.haversine_distance(0.0, 0.0, 0.0, 1.0)
    assert abs(d - 111.19) < 1.0
    poly = [(0,0), (0,4), (4,4), (4,0)]
    assert GISSpatialEngine.point_in_polygon(2, 2, poly) is True
    assert GISSpatialEngine.point_in_polygon(5, 5, poly) is False
