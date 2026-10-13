import pytest
import numpy as np
from engineering_continuum.field_420_electro_tactile_lingual_rasterizer.lingual_tactile_rasterizer import LingualTactileMatrixRasterizer

def test_field_420_initialization():
    rasterizer = LingualTactileMatrixRasterizer(grid_size=8)
    assert rasterizer.grid_size == 8
    assert rasterizer.active_matrix.shape == (8, 8)

def test_field_420_polar_depth_mapping():
    rasterizer = LingualTactileMatrixRasterizer(grid_size=8)
    # Forward close point (0 deg, 1.0 m)
    pts = [(0.0, 1.0, 1.0)]
    grid = rasterizer.rasterize_depth_points(pts)
    
    # Forward point maps to upper center (low y, middle x)
    assert rasterizer.get_tactile_energy() > 0.0
    assert bool(rasterizer.verify_properties()) is True
