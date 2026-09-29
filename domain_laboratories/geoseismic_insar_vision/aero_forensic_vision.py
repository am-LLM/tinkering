"""
Aero Forensic Vision Engine:
1. 3D Multi-Camera Photogrammetric Point-Cloud Triangulation (DLT / SVD).
2. Aerodynamic Ballistics Trajectory Simulator (Nonlinear Drag + Barometric Density + Coriolis Deflection).
3. Bloodstain Pattern Analysis (BPA) 3D Area of Origin Fluid Solver.

Mathematical Formulations:
---------------------------
- Photogrammetry DLT: A X = 0 where A_i = [u_i * P_i^(3) - P_i^(1); v_i * P_i^(3) - P_i^(2)]
- Ballistic Dynamics: m * dv/dt = m*g - 0.5 * rho(z) * Cd * A * |v - w| * (v - w) - 2*m*(Omega x v)
- BPA Point of Origin: Weighted orthogonal distance minimization across 3D directional vectors:
  ( sum_i w_i * (I - u_i * u_i^T) ) * p_origin = sum_i w_i * (I - u_i * u_i^T) * p_i
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


# =====================================================================
# 1. 3D PHOTOGRAMMETRIC TRIANGULATION
# =====================================================================

@dataclass
class CameraParameters:
    """Pinhole Camera Intrinsics and Extrinsics."""
    camera_id: int
    focal_length_px: Tuple[float, float]  # (fx, fy)
    principal_point_px: Tuple[float, float] # (cx, cy)
    rotation_matrix: np.ndarray             # R (3x3)
    translation_vector: np.ndarray          # t (3,)

    @property
    def intrinsic_matrix_k(self) -> np.ndarray:
        """3x3 Intrinsic matrix K."""
        fx, fy = self.focal_length_px
        cx, cy = self.principal_point_px
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @property
    def projection_matrix_p(self) -> np.ndarray:
        """3x4 Projection matrix P = K * [R | t]."""
        k = self.intrinsic_matrix_k
        rt = np.hstack([self.rotation_matrix, self.translation_vector.reshape(3, 1)])
        return np.dot(k, rt)

    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
        """Projects 3D point in world coordinates to 2D image plane (u, v)."""
        x_homog = np.append(point_3d, 1.0)
        p_homog = np.dot(self.projection_matrix_p, x_homog)
        if abs(p_homog[2]) < 1e-12:
            return np.array([0.0, 0.0])
        return np.array([p_homog[0] / p_homog[2], p_homog[1] / p_homog[2]])


class PhotogrammetricTriangulator:
    """
    Solves 3D point cloud triangulation from multi-view 2D camera observations.
    """

    @staticmethod
    def triangulate_point_dlt(
        cameras: List[CameraParameters],
        observations_2d: List[np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """
        Direct Linear Transform (DLT) using Singular Value Decomposition (SVD).
        Returns:
            (point_3d, reprojection_rmse)
        """
        if len(cameras) < 2 or len(observations_2d) != len(cameras):
            raise ValueError("At least 2 camera observations are required for triangulation.")

        num_cams = len(cameras)
        a_matrix = np.zeros((2 * num_cams, 4), dtype=np.float64)

        for i, (cam, uv) in enumerate(zip(cameras, observations_2d)):
            p = cam.projection_matrix_p
            u, v = uv[0], uv[1]
            a_matrix[2 * i]     = u * p[2, :] - p[0, :]
            a_matrix[2 * i + 1] = v * p[2, :] - p[1, :]

        # SVD solution: A * X = 0
        _, _, vh = np.linalg.svd(a_matrix)
        x_homog = vh[-1]
        
        if abs(x_homog[3]) < 1e-15:
            point_3d = x_homog[:3]
        else:
            point_3d = x_homog[:3] / x_homog[3]

        # Calculate root-mean-square reprojection error
        reproj_errors = []
        for cam, uv in zip(cameras, observations_2d):
            uv_pred = cam.project_point(point_3d)
            reproj_errors.append(np.linalg.norm(uv - uv_pred))

        rmse = float(np.sqrt(np.mean(np.array(reproj_errors) ** 2)))
        return point_3d, rmse

    @classmethod
    def triangulate_point_cloud(
        cls,
        cameras: List[CameraParameters],
        multi_view_points_2d: List[List[np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Triangulates an entire batch of point correspondences across multiple views.
        """
        points_3d = []
        rmses = []
        for obs_set in multi_view_points_2d:
            pt3d, err = cls.triangulate_point_dlt(cameras, obs_set)
            points_3d.append(pt3d)
            rmses.append(err)
        return np.array(points_3d), np.array(rmses)


# =====================================================================
# 2. AERODYNAMIC BALLISTIC TRAJECTORY SIMULATION
# =====================================================================

@dataclass
class ProjectileProperties:
    """Physical characteristics of ballistic projectile."""
    mass_kg: float = 0.009           # 9mm Parabellum bullet (~9g)
    cross_section_area_m2: float = 6.36e-5 # Caliber ~9mm (pi * (0.0045)^2)
    drag_coefficient_cd: float = 0.295     # Supersonic/transonic form factor
    drag_model: str = "quadratic"          # Quadratic aerodynamic drag


@dataclass
class BallisticEnvironment:
    """Atmospheric and planetary geophysical environment."""
    sea_level_density_kg_m3: float = 1.225
    scale_height_m: float = 8500.0          # Atmospheric scale height for density decay
    gravity_mps2: float = 9.80665
    wind_velocity_mps: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    latitude_rad: float = np.deg2rad(45.0)   # Geographic latitude for Coriolis force
    earth_angular_velocity_rad_s: float = 7.292115e-5

    def air_density(self, altitude_m: float) -> float:
        """Barometric exponential atmospheric density model."""
        alt = max(altitude_m, 0.0)
        return self.sea_level_density_kg_m3 * np.exp(-alt / self.scale_height_m)

    @property
    def earth_omega_vector(self) -> np.ndarray:
        """Earth angular velocity vector in local topocentric frame (East, North, Up)."""
        # In ENU frame: Omega = (0, Omega * cos(phi), Omega * sin(phi))
        return np.array([
            0.0,
            self.earth_angular_velocity_rad_s * np.cos(self.latitude_rad),
            self.earth_angular_velocity_rad_s * np.sin(self.latitude_rad)
        ], dtype=np.float64)


@dataclass
class TrajectoryOutput:
    """Ballistic trajectory state integration records."""
    time_s: np.ndarray
    positions_m: np.ndarray    # Shape: (N, 3) in [East, North, Up]
    velocities_mps: np.ndarray # Shape: (N, 3)
    flight_time_s: float
    maximum_range_m: float
    apogee_altitude_m: float
    impact_velocity_mps: float
    coriolis_deflection_m: float


class AerodynamicBallisticsSimulator:
    """
    Solves 3D nonlinear ballistic ODEs with drag, atmospheric lapse, and Coriolis deflection.
    """

    def __init__(
        self,
        projectile: Optional[ProjectileProperties] = None,
        environment: Optional[BallisticEnvironment] = None
    ):
        self.proj = projectile or ProjectileProperties()
        self.env = environment or BallisticEnvironment()

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """
        state = [x, y, z, vx, vy, vz]
        dstate/dt = [vx, vy, vz, ax, ay, az]
        """
        pos = state[:3]
        vel = state[3:]
        z = pos[2]

        # Relative velocity with wind
        v_rel = vel - self.env.wind_velocity_mps
        speed_rel = np.linalg.norm(v_rel)

        # Aerodynamic drag acceleration: a_drag = -0.5 * rho * Cd * A / m * |v_rel| * v_rel
        rho = self.env.air_density(z)
        drag_factor = 0.5 * rho * self.proj.drag_coefficient_cd * self.proj.cross_section_area_m2 / self.proj.mass_kg
        a_drag = - drag_factor * speed_rel * v_rel

        # Gravity acceleration: a_g = [0, 0, -g]
        a_grav = np.array([0.0, 0.0, -self.env.gravity_mps2])

        # Coriolis acceleration: a_coriolis = -2 * (Omega x v)
        omega = self.env.earth_omega_vector
        a_coriolis = -2.0 * np.cross(omega, vel)

        total_acc = a_drag + a_grav + a_coriolis
        return np.hstack([vel, total_acc])

    def simulate(
        self,
        muzzle_velocity_mps: float = 380.0,
        elevation_angle_deg: float = 45.0,
        azimuth_angle_deg: float = 0.0, # 0 = North, 90 = East
        initial_pos_m: Optional[np.ndarray] = None,
        dt_s: float = 0.005,
        max_time_s: float = 120.0
    ) -> TrajectoryOutput:
        """
        Integrates ballistic trajectory until ground impact (z <= 0).
        """
        pos0 = initial_pos_m if initial_pos_m is not None else np.array([0.0, 0.0, 1.5])
        
        # Convert elevation and azimuth to velocity vector in ENU coordinates
        elev_rad = np.deg2rad(elevation_angle_deg)
        azim_rad = np.deg2rad(azimuth_angle_deg)

        vx0 = muzzle_velocity_mps * np.cos(elev_rad) * np.sin(azim_rad) # East
        vy0 = muzzle_velocity_mps * np.cos(elev_rad) * np.cos(azim_rad) # North
        vz0 = muzzle_velocity_mps * np.sin(elev_rad)                     # Up
        
        state = np.hstack([pos0, [vx0, vy0, vz0]])
        
        time_hist = [0.0]
        pos_hist = [pos0.copy()]
        vel_hist = [state[3:].copy()]

        t = 0.0
        while t < max_time_s:
            # 4th-Order Runge-Kutta (RK4) Step
            k1 = self._derivatives(state)
            k2 = self._derivatives(state + 0.5 * dt_s * k1)
            k3 = self._derivatives(state + 0.5 * dt_s * k2)
            k4 = self._derivatives(state + dt_s * k3)

            state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += dt_s

            time_hist.append(t)
            pos_hist.append(state[:3].copy())
            vel_hist.append(state[3:].copy())

            # Check ground impact
            if state[2] <= 0.0:
                break

        positions = np.array(pos_hist)
        velocities = np.array(vel_hist)
        times = np.array(time_hist)

        # Compute trajectory summary metrics
        flight_time = times[-1]
        max_range = float(np.linalg.norm(positions[-1, :2] - positions[0, :2]))
        apogee = float(np.max(positions[:, 2]))
        impact_vel = float(np.linalg.norm(velocities[-1]))

        # Coriolis cross-track deflection calculation
        launch_dir_2d = np.array([np.sin(azim_rad), np.cos(azim_rad)])
        cross_dir_2d = np.array([np.cos(azim_rad), -np.sin(azim_rad)])
        final_disp_2d = positions[-1, :2] - positions[0, :2]
        coriolis_deflection = float(abs(np.dot(final_disp_2d, cross_dir_2d)))

        return TrajectoryOutput(
            time_s=times,
            positions_m=positions,
            velocities_mps=velocities,
            flight_time_s=flight_time,
            maximum_range_m=max_range,
            apogee_altitude_m=apogee,
            impact_velocity_mps=impact_vel,
            coriolis_deflection_m=coriolis_deflection
        )


# =====================================================================
# 3. BLOODSTAIN PATTERN ANALYSIS (BPA) FLUID ORIGIN SOLVER
# =====================================================================

@dataclass
class BloodstainRecord:
    """Morphometric measurement of an elliptical bloodstain on a surface."""
    stain_id: int
    surface_pos_3d: np.ndarray  # Impact coordinate on target surface (x, y, z)
    major_axis_mm: float        # Ellipse major diameter l
    minor_axis_mm: float        # Ellipse minor diameter w (w <= l)
    gamma_direction_deg: float  # In-plane directional angle (degrees) relative to horizontal tangent
    surface_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0])) # Surface normal vector

    @property
    def impact_angle_rad(self) -> float:
        """Balthazard Impact Angle alpha = arcsin(w / l)."""
        ratio = np.clip(self.minor_axis_mm / max(self.major_axis_mm, 1e-6), 0.0, 1.0)
        return float(np.arcsin(ratio))

    @property
    def trajectory_direction_3d(self) -> np.ndarray:
        """
        Unit vector pointing backwards along blood droplet incoming flight trajectory.
        """
        alpha = self.impact_angle_rad
        gamma_rad = np.deg2rad(self.gamma_direction_deg)

        # Construct consistent orthonormal coordinate frame on surface
        n = self.surface_normal / np.linalg.norm(self.surface_normal)
        
        # Horizontal tangent t1 and vertical/elevation tangent t2
        if abs(n[2]) < 0.9:
            # Vertical or slanted wall
            # t1 along horizontal: cross([0, 0, 1], n)
            t1 = np.array([n[1], -n[0], 0.0])
            norm_t1 = np.linalg.norm(t1)
            if norm_t1 > 1e-6:
                t1 = t1 / norm_t1
            else:
                t1 = np.array([1.0, 0.0, 0.0])
            # For n = [0, -1, 0], t1 becomes [1, 0, 0] if we sign correctly
            if np.dot(t1, np.array([1.0, 0.0, 0.0])) < 0:
                t1 = -t1
            t2 = np.cross(n, t1)
            if t2[2] < 0:
                t2 = -t2
        else:
            # Floor or ceiling
            t1 = np.array([1.0, 0.0, 0.0])
            t2 = np.array([0.0, 1.0, 0.0])

        # In-plane droplet velocity direction: v_plane = cos(gamma)*t1 + sin(gamma)*t2
        v_plane = np.cos(gamma_rad) * t1 + np.sin(gamma_rad) * t2
        
        # Backwards trajectory direction vector:
        # Opposite to in-plane direction, pointing away from surface (+n)
        u_dir = - np.cos(alpha) * v_plane + np.sin(alpha) * n
        return u_dir / np.linalg.norm(u_dir)


@dataclass
class BPAOriginResult:
    """Estimated 3D point/area of origin and statistical convergence confidence."""
    origin_point_3d: np.ndarray     # (X, Y, Z) in meters
    residual_rmse_m: float           # Root Mean Square orthogonal distance error
    stain_weights: np.ndarray
    number_of_stains_used: int
    uncertainty_radius_m: float


class BloodstainFluidOriginSolver:
    """
    Reconstructs 3D Area of Origin (PO / AO) from spatter patterns.
    Uses weighted orthogonal ray distance minimization.
    """

    @staticmethod
    def solve_origin_3d(
        stains: List[BloodstainRecord],
        min_impact_angle_deg: float = 10.0,
        max_impact_angle_deg: float = 80.0
    ) -> BPAOriginResult:
        """
        Finds 3D intersection origin point minimizing perpendicular distances to trajectory rays:
        min_p sum_i w_i * || (p - p_i) - ((p - p_i) . u_i) * u_i ||^2
        """
        valid_stains = []
        for s in stains:
            angle_deg = np.rad2deg(s.impact_angle_rad)
            if min_impact_angle_deg <= angle_deg <= max_impact_angle_deg:
                valid_stains.append(s)

        if len(valid_stains) < 3:
            raise ValueError(f"Need at least 3 valid bloodstains with reliable impact angles, found {len(valid_stains)}.")

        n_stains = len(valid_stains)
        m_matrix = np.zeros((3, 3), dtype=np.float64)
        rhs_vector = np.zeros(3, dtype=np.float64)
        weights = np.zeros(n_stains, dtype=np.float64)

        for i, s in enumerate(valid_stains):
            p_i = s.surface_pos_3d
            u_i = s.trajectory_direction_3d
            
            # Geometric weight factor (sin*cos is maximal near 45 deg)
            angle_rad = s.impact_angle_rad
            w_i = float(np.sin(angle_rad) * np.cos(angle_rad))
            weights[i] = w_i

            # Orthogonal projection operator
            p_perp = np.eye(3) - np.outer(u_i, u_i)

            m_matrix += w_i * p_perp
            rhs_vector += w_i * np.dot(p_perp, p_i)

        # Regularized solve for 3D origin point
        m_matrix += 1e-12 * np.eye(3)
        origin_pt = np.linalg.solve(m_matrix, rhs_vector)

        # Compute orthogonal residual distances
        residuals = []
        for s in valid_stains:
            p_i = s.surface_pos_3d
            u_i = s.trajectory_direction_3d
            diff = origin_pt - p_i
            ortho_dist = np.linalg.norm(diff - np.dot(diff, u_i) * u_i)
            residuals.append(ortho_dist)

        rmse = float(np.sqrt(np.mean(np.array(residuals) ** 2)))
        uncertainty_r = float(np.std(residuals) * 1.96) # 95% confidence radius

        return BPAOriginResult(
            origin_point_3d=origin_pt,
            residual_rmse_m=rmse,
            stain_weights=weights,
            number_of_stains_used=n_stains,
            uncertainty_radius_m=uncertainty_r
        )
