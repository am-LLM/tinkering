"""Course 186: 3D Image Affine Transformation & Trilinear Voxel Interpolator"""
import numpy as np

class VoxelTrilinearInterpolator:
    @staticmethod
    def interpolate_point(volume: np.ndarray, x: float, y: float, z: float) -> float:
        x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        x1, y1, z1 = min(volume.shape[0]-1, x0 + 1), min(volume.shape[1]-1, y0 + 1), min(volume.shape[2]-1, z0 + 1)
        
        xd, yd, zd = x - x0, y - y0, z - z0
        
        c000 = volume[x0, y0, z0]
        c100 = volume[x1, y0, z0]
        c010 = volume[x0, y1, z0]
        c110 = volume[x1, y1, z0]
        c001 = volume[x0, y0, z1]
        c101 = volume[x1, y0, z1]
        c011 = volume[x0, y1, z1]
        c111 = volume[x1, y1, z1]
        
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        
        return float(c0 * (1 - zd) + c1 * zd)
