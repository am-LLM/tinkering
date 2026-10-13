"""Course 420: Electro-Tactile Lingual Sensory Substitution Matrix Rasterizer"""
import numpy as np

class LingualTactileMatrixRasterizer:
    def __init__(self, grid_size: int = 8, min_freq_hz: float = 20.0, max_freq_hz: float = 400.0):
        self.grid_size = grid_size
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.active_matrix = np.zeros((grid_size, grid_size), dtype=float)

    def rasterize_depth_points(self, point_cloud_polar: list) -> np.ndarray:
        """
        Converts list of (azimuth_deg, distance_m, intensity) into grid frequencies.
        Azimuth: 0 deg = Forward, 90 deg = Right, 180 deg = Rear, 270 deg = Left.
        Distance: Nearer obstacles produce higher tactile stimulation frequency.
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=float)
        center = (self.grid_size - 1) / 2.0
        
        for pt in point_cloud_polar:
            azimuth, dist, intensity = pt[0], pt[1], pt[2]
            az_rad = np.radians(azimuth)
            
            # Distance mapping: closer = higher normalized proximity (1.0 = right at skin)
            proximity = float(np.clip(1.0 - (dist / 10.0), 0.0, 1.0))
            if proximity <= 0.0:
                continue
                
            # Polar to 2D matrix coordinates
            r = proximity * center
            x = int(np.clip(round(center + r * np.sin(az_rad)), 0, self.grid_size - 1))
            y = int(np.clip(round(center - r * np.cos(az_rad)), 0, self.grid_size - 1))
            
            # Map proximity and intensity to electrical stimulation pulse frequency (Hz)
            freq = self.min_freq_hz + (proximity * intensity) * (self.max_freq_hz - self.min_freq_hz)
            grid[y, x] = max(grid[y, x], freq)
            
        self.active_matrix = grid
        return grid

    def get_tactile_energy(self) -> float:
        return float(np.sum(self.active_matrix))

    def verify_properties(self) -> bool:
        test_pts = [(0.0, 1.0, 1.0), (180.0, 2.0, 0.8)]
        mat = self.rasterize_depth_points(test_pts)
        return bool(np.all(np.isfinite(mat)) and np.all(mat >= 0.0))
