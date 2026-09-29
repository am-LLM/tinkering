import numpy as np
from voxel_trilinear_interpolator import VoxelTrilinearInterpolator

def test_voxel_interp():
    vol = np.zeros((4, 4, 4))
    vol[1, 1, 1] = 10.0
    val = VoxelTrilinearInterpolator.interpolate_point(vol, 1.0, 1.0, 1.0)
    assert abs(val - 10.0) < 1e-4
