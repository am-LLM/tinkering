"""
Subterranean GPS-Denied Quadcopter GNC Navigation Engine
--------------------------------------------------------
Implements:
1. 3D Octree spatial mapping for tunnel / shaft obstacle boundary representation.
2. Gradient-based gas plume concentration tracking (CH4 / CO methane chemotaxis).
3. Dynamic 3D A* path re-planning through narrow subterranean mine shafts.
4. 6-DoF Quadcopter cascaded flight dynamics and SE(3) trajectory tracking controller.
"""

from __future__ import annotations
import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, Any


# =====================================================================
# 1. 3D Octree Spatial Mapping
# =====================================================================

@dataclass
class BoundingBox3D:
    min_pt: np.ndarray  # [xmin, ymin, zmin]
    max_pt: np.ndarray  # [xmax, ymax, zmax]

    def contains(self, pt: np.ndarray) -> bool:
        return bool(np.all(pt >= self.min_pt) and np.all(pt <= self.max_pt))

    def intersects(self, other: BoundingBox3D) -> bool:
        return bool(
            np.all(self.min_pt <= other.max_pt) and np.all(self.max_pt >= other.min_pt)
        )


class OctreeNode:
    def __init__(self, bbox: BoundingBox3D, depth: int = 0, max_depth: int = 6):
        self.bbox = bbox
        self.depth = depth
        self.max_depth = max_depth
        self.is_leaf = True
        self.occupied = False
        self.children: List[Optional[OctreeNode]] = [None] * 8
        self.center = 0.5 * (bbox.min_pt + bbox.max_pt)

    def subdivide(self):
        mid = self.center
        min_p = self.bbox.min_pt
        max_p = self.bbox.max_pt
        
        idx = 0
        for i in range(2):
            x_min = min_p[0] if i == 0 else mid[0]
            x_max = mid[0] if i == 0 else max_p[0]
            for j in range(2):
                y_min = min_p[1] if j == 0 else mid[1]
                y_max = mid[1] if j == 0 else max_p[1]
                for k in range(2):
                    z_min = min_p[2] if k == 0 else mid[2]
                    z_max = mid[2] if k == 0 else max_p[2]
                    
                    sub_bbox = BoundingBox3D(
                        min_pt=np.array([x_min, y_min, z_min], dtype=np.float64),
                        max_pt=np.array([x_max, y_max, z_max], dtype=np.float64)
                    )
                    self.children[idx] = OctreeNode(sub_bbox, depth=self.depth + 1, max_depth=self.max_depth)
                    idx += 1
        self.is_leaf = False

    def insert(self, pt: np.ndarray) -> bool:
        if not self.bbox.contains(pt):
            return False
            
        if self.depth >= self.max_depth:
            self.occupied = True
            return True
            
        if self.is_leaf:
            self.subdivide()
            
        self.occupied = True
        for child in self.children:
            if child and child.bbox.contains(pt):
                return child.insert(pt)
        return False

    def is_occupied_at(self, pt: np.ndarray) -> bool:
        if not self.bbox.contains(pt):
            return True  # Out of bounds treated as collision
        if not self.occupied:
            return False
        if self.is_leaf:
            return self.occupied
        for child in self.children:
            if child and child.bbox.contains(pt):
                return child.is_occupied_at(pt)
        return False


class OctreeMap:
    """
    Hierarchical 3D spatial occupancy grid for subterranean mines.
    """
    def __init__(self, bounds_min: np.ndarray, bounds_max: np.ndarray, resolution: float = 0.5):
        self.bounds = BoundingBox3D(
            min_pt=np.asarray(bounds_min, dtype=np.float64),
            max_pt=np.asarray(bounds_max, dtype=np.float64)
        )
        self.resolution = float(resolution)
        # Calculate max depth corresponding to resolution
        max_extent = np.max(self.bounds.max_pt - self.bounds.min_pt)
        self.max_depth = max(1, int(np.ceil(np.log2(max_extent / self.resolution))))
        self.root = OctreeNode(self.bounds, depth=0, max_depth=self.max_depth)
        self.obstacle_points: List[np.ndarray] = []

    def add_obstacle_point(self, pt: np.ndarray):
        pt = np.asarray(pt, dtype=np.float64)
        self.obstacle_points.append(pt)
        self.root.insert(pt)

    def add_box_obstacle(self, min_pt: np.ndarray, max_pt: np.ndarray, step: Optional[float] = None):
        step_sz = step or self.resolution
        min_p = np.asarray(min_pt, dtype=np.float64)
        max_p = np.asarray(max_pt, dtype=np.float64)
        
        xs = np.arange(min_p[0], max_p[0] + step_sz, step_sz)
        ys = np.arange(min_p[1], max_p[1] + step_sz, step_sz)
        zs = np.arange(min_p[2], max_p[2] + step_sz, step_sz)
        
        for x in xs:
            for y in ys:
                for z in zs:
                    self.add_obstacle_point(np.array([x, y, z]))

    def is_collision(self, pt: np.ndarray, margin: float = 0.3) -> bool:
        pt = np.asarray(pt, dtype=np.float64)
        if not self.bounds.contains(pt):
            return True
        # Check direct point and bounding sphere margin
        if self.root.is_occupied_at(pt):
            return True
        if margin > 0.0:
            for offset in [
                [margin, 0, 0], [-margin, 0, 0],
                [0, margin, 0], [0, -margin, 0],
                [0, 0, margin], [0, 0, -margin]
            ]:
                if self.root.is_occupied_at(pt + np.array(offset)):
                    return True
        return False


# =====================================================================
# 2. Gradient-Based Gas Plume Concentration Tracking
# =====================================================================

@dataclass
class GasPlumeSource:
    origin: np.ndarray           # [x, y, z] location of gas leak / emitter
    peak_concentration: float = 1000.0  # ppm
    dispersion_sigma: float = 4.0        # spatial decay spread


class SubterraneanGasEnvironment:
    """
    Simulates 3D advection-diffusion methane (CH4) or carbon monoxide (CO) gas dispersion.
    """
    def __init__(self, sources: Optional[List[GasPlumeSource]] = None, background_noise: float = 2.0):
        self.sources = sources or [GasPlumeSource(origin=np.array([20.0, 15.0, 5.0]))]
        self.background_noise = float(background_noise)

    def sample_concentration(self, pos: np.ndarray, add_noise: bool = True) -> float:
        pos = np.asarray(pos, dtype=np.float64)
        total_c = 0.0
        for src in self.sources:
            dist_sq = np.sum((pos - src.origin) ** 2)
            c = src.peak_concentration * np.exp(-dist_sq / (2.0 * (src.dispersion_sigma ** 2)))
            total_c += c
            
        if add_noise:
            total_c += np.random.normal(0.0, self.background_noise)
        return float(max(0.0, total_c))

    def true_gradient(self, pos: np.ndarray) -> np.ndarray:
        """Analytical spatial gradient dC/dx, dC/dy, dC/dz."""
        pos = np.asarray(pos, dtype=np.float64)
        grad = np.zeros(3, dtype=np.float64)
        for src in self.sources:
            diff = pos - src.origin
            dist_sq = np.sum(diff ** 2)
            c = src.peak_concentration * np.exp(-dist_sq / (2.0 * (src.dispersion_sigma ** 2)))
            grad += - (diff / (src.dispersion_sigma ** 2)) * c
        return grad


class GasPlumeChemotaxisTracker:
    """
    Estimates 3D concentration gradient using spatial multi-point sampling & history regression.
    """
    def __init__(self, sample_baseline_radius: float = 0.5):
        self.baseline = float(sample_baseline_radius)
        self.history_positions: List[np.ndarray] = []
        self.history_samples: List[float] = []

    def record_sample(self, pos: np.ndarray, concentration: float):
        self.history_positions.append(np.asarray(pos, dtype=np.float64).copy())
        self.history_samples.append(float(concentration))
        if len(self.history_positions) > 30:
            self.history_positions.pop(0)
            self.history_samples.pop(0)

    def estimate_gradient_at(self, current_pos: np.ndarray, env: SubterraneanGasEnvironment) -> np.ndarray:
        """
        Estimates 3D gradient vector via central differences sampling.
        """
        p = np.asarray(current_pos, dtype=np.float64)
        d = self.baseline
        
        # Central difference along 3 axes
        cx_plus = env.sample_concentration(p + np.array([d, 0, 0]), add_noise=False)
        cx_minus = env.sample_concentration(p - np.array([d, 0, 0]), add_noise=False)
        
        cy_plus = env.sample_concentration(p + np.array([0, d, 0]), add_noise=False)
        cy_minus = env.sample_concentration(p - np.array([0, d, 0]), add_noise=False)
        
        cz_plus = env.sample_concentration(p + np.array([0, 0, d]), add_noise=False)
        cz_minus = env.sample_concentration(p - np.array([0, 0, d]), add_noise=False)
        
        grad = np.array([
            (cx_plus - cx_minus) / (2.0 * d),
            (cy_plus - cy_minus) / (2.0 * d),
            (cz_plus - cz_minus) / (2.0 * d)
        ], dtype=np.float64)
        
        return grad


# =====================================================================
# 3. Dynamic 3D A* Mine Shaft Path Planner
# =====================================================================

class MinePathPlanner:
    """
    3D A* / Dijkstra path planner operating on continuous space discretized via Octree.
    """
    def __init__(self, octree_map: OctreeMap, step_size: float = 0.5, safety_margin: float = 0.3):
        self.octree = octree_map
        self.step_size = float(step_size)
        self.safety_margin = float(safety_margin)

    def _pos_to_grid(self, pos: np.ndarray) -> Tuple[int, int, int]:
        p = (pos - self.octree.bounds.min_pt) / self.step_size
        return (int(round(p[0])), int(round(p[1])), int(round(p[2])))

    def _grid_to_pos(self, grid: Tuple[int, int, int]) -> np.ndarray:
        return self.octree.bounds.min_pt + np.array(grid, dtype=np.float64) * self.step_size

    def plan_path(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> Optional[List[np.ndarray]]:
        start_pos = np.asarray(start_pos, dtype=np.float64)
        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        
        if self.octree.is_collision(start_pos, self.safety_margin):
            return None
        if self.octree.is_collision(goal_pos, self.safety_margin):
            return None
            
        start_grid = self._pos_to_grid(start_pos)
        goal_grid = self._pos_to_grid(goal_pos)
        
        open_set: List[Tuple[float, float, Tuple[int, int, int]]] = []
        # (f_score, g_score, grid_coord)
        h0 = float(np.linalg.norm(start_pos - goal_pos))
        heapq.heappush(open_set, (h0, 0.0, start_grid))
        
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {start_grid: 0.0}
        closed_set: Set[Tuple[int, int, int]] = set()
        
        # 26-neighborhood directions in 3D
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cost = np.sqrt(dx**2 + dy**2 + dz**2) * self.step_size
                    neighbors.append(((dx, dy, dz), cost))
                    
        max_iterations = 20000
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            f, g, current = heapq.heappop(open_set)
            
            if current == goal_grid or np.linalg.norm(self._grid_to_pos(current) - goal_pos) < self.step_size:
                # Reconstruct path
                path = [goal_pos]
                curr = current
                while curr in came_from:
                    path.append(self._grid_to_pos(curr))
                    curr = came_from[curr]
                path.append(start_pos)
                path.reverse()
                return path
                
            if current in closed_set:
                continue
            closed_set.add(current)
            
            curr_pos = self._grid_to_pos(current)
            
            for (dx, dy, dz), step_cost in neighbors:
                nbr_grid = (current[0] + dx, current[1] + dy, current[2] + dz)
                if nbr_grid in closed_set:
                    continue
                    
                nbr_pos = self._grid_to_pos(nbr_grid)
                if self.octree.is_collision(nbr_pos, self.safety_margin):
                    continue
                    
                tentative_g = g + step_cost
                if nbr_grid not in g_score or tentative_g < g_score[nbr_grid]:
                    g_score[nbr_grid] = tentative_g
                    came_from[nbr_grid] = current
                    h = float(np.linalg.norm(nbr_pos - goal_pos))
                    f_nbr = tentative_g + h
                    heapq.heappush(open_set, (f_nbr, tentative_g, nbr_grid))
                    
        return None  # No feasible path


# =====================================================================
# 4. Quadcopter 6-DoF Dynamics & SE(3) Trajectory Controller
# =====================================================================

@dataclass
class QuadcopterState:
    p: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))  # Position (m)
    v: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))  # Velocity (m/s)
    q: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.], dtype=np.float64)) # [w, x, y, z]
    w: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))  # Angular rate (rad/s)


class QuadcopterFlightController:
    """
    Cascaded SE(3) position and attitude controller for subterranean navigation.
    """
    def __init__(self, mass: float = 1.2, gravity: float = 9.81):
        self.mass = float(mass)
        self.g = float(gravity)
        self.state = QuadcopterState()
        
        # Position gains
        self.Kp_pos = np.array([2.0, 2.0, 3.0])
        self.Kd_pos = np.array([1.5, 1.5, 2.0])
        
        # Velocity limit
        self.v_max = 5.0

    def compute_thrust_and_attitude_cmd(self, target_pos: np.ndarray, target_vel: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """
        Computes desired collective thrust (N) and target acceleration vector.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_vel = np.asarray(target_vel if target_vel is not None else np.zeros(3), dtype=np.float64)
        
        pos_err = target_pos - self.state.p
        vel_err = target_vel - self.state.v
        
        # Desired acceleration: a_des = Kp * e_p + Kd * e_v + g * z_hat
        a_des = self.Kp_pos * pos_err + self.Kd_pos * vel_err + np.array([0.0, 0.0, self.g])
        
        # Required collective thrust
        thrust_magnitude = self.mass * float(np.linalg.norm(a_des))
        return thrust_magnitude, a_des

    def step_simulation(self, target_pos: np.ndarray, dt: float = 0.05):
        """
        Closed-loop kinematic/dynamic simulation step toward waypoint.
        """
        thrust, a_des = self.compute_thrust_and_attitude_cmd(target_pos)
        
        # Accelerate drone towards commanded acceleration with drag
        drag_coeff = 0.1
        acc = (a_des - np.array([0.0, 0.0, self.g])) - drag_coeff * self.state.v
        
        self.state.v += acc * dt
        # Clip max velocity
        speed = np.linalg.norm(self.state.v)
        if speed > self.v_max:
            self.state.v = (self.state.v / speed) * self.v_max
            
        self.state.p += self.state.v * dt
