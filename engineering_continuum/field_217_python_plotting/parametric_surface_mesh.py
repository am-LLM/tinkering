"""Course 217: 3D Parametric Torus & Sphere Surface Mesh Generator"""
import numpy as np

class ParametricSurfaceMesh:
    @staticmethod
    def generate_torus(r_major: float = 3.0, r_minor: float = 1.0, num_u: int = 10, num_v: int = 10) -> tuple:
        u = np.linspace(0, 2*np.pi, num_u)
        v = np.linspace(0, 2*np.pi, num_v)
        u_grid, v_grid = np.meshgrid(u, v)
        
        x = (r_major + r_minor * np.cos(v_grid)) * np.cos(u_grid)
        y = (r_major + r_minor * np.cos(v_grid)) * np.sin(u_grid)
        z = r_minor * np.sin(v_grid)
        return x, y, z
