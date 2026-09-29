from parametric_surface_mesh import ParametricSurfaceMesh

def test_torus_mesh():
    x, y, z = ParametricSurfaceMesh.generate_torus(3.0, 1.0, 10, 10)
    assert x.shape == (10, 10)
    assert y.shape == (10, 10)
    assert z.shape == (10, 10)
