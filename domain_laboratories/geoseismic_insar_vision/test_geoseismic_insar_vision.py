"""
Extended OPFOR Verification & Deep Mathematical Test Suite for Cluster 9:
1. aero_forensic_vision.py
2. geo_seismic_inversion.py
3. insar_sentinel_pipeline.py
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from aero_forensic_vision import (
    CameraParameters,
    PhotogrammetricTriangulator,
    ProjectileProperties,
    BallisticEnvironment,
    AerodynamicBallisticsSimulator,
    BloodstainRecord,
    BloodstainFluidOriginSolver
)

from geo_seismic_inversion import (
    SeismicGridConfig,
    SeismicSurveyGeometry,
    FullWaveformInversion2D,
    FWIInversionResult
)

from insar_sentinel_pipeline import (
    SENTINEL1_WAVELENGTH_M,
    SARAcquisition,
    InterferogramPair,
    SentinelInSARPipeline,
    InSARDeformationResult
)


# =====================================================================
# 1. AERO FORENSIC VISION TESTS
# =====================================================================
class TestAeroForensicVision:
    def test_camera_projection_and_triangulation(self):
        target_point = np.array([1.5, 2.0, 5.0])
        
        r1, t1 = np.eye(3), np.array([0.0, 0.0, 0.0])
        cam1 = CameraParameters(1, (1000.0, 1000.0), (500.0, 500.0), r1, t1)
        
        r2, t2 = np.eye(3), np.array([-1.0, 0.0, 0.0])
        cam2 = CameraParameters(2, (1000.0, 1000.0), (500.0, 500.0), r2, t2)

        uv1 = cam1.project_point(target_point)
        uv2 = cam2.project_point(target_point)

        triangulator = PhotogrammetricTriangulator()
        recon_pt, rmse = triangulator.triangulate_point_dlt([cam1, cam2], [uv1, uv2])

        assert np.allclose(recon_pt, target_point, atol=1e-5)
        assert rmse < 1e-4

    def test_triangulation_minimum_camera_validation(self):
        r1, t1 = np.eye(3), np.zeros(3)
        cam1 = CameraParameters(1, (800.0, 800.0), (400.0, 400.0), r1, t1)
        with pytest.raises(ValueError, match="At least 2 camera observations"):
            PhotogrammetricTriangulator.triangulate_point_dlt([cam1], [np.array([100.0, 100.0])])

    def test_point_cloud_batch_triangulation(self):
        r1, t1 = np.eye(3), np.zeros(3)
        r2, t2 = np.eye(3), np.array([-2.0, 0.0, 0.0])
        cam1 = CameraParameters(1, (800.0, 800.0), (400.0, 400.0), r1, t1)
        cam2 = CameraParameters(2, (800.0, 800.0), (400.0, 400.0), r2, t2)

        pts_true = [np.array([0.5, -0.5, 4.0]), np.array([-0.2, 1.0, 6.0]), np.array([1.0, 0.0, 3.5])]
        obs_batch = [[cam1.project_point(p), cam2.project_point(p)] for p in pts_true]

        recon_pts, rmses = PhotogrammetricTriangulator.triangulate_point_cloud([cam1, cam2], obs_batch)
        assert len(recon_pts) == 3
        assert np.allclose(recon_pts, pts_true, atol=1e-4)

    def test_vacuum_ballistics_analytical_comparison(self):
        proj_vacuum = ProjectileProperties(drag_coefficient_cd=0.0)
        env_vacuum = BallisticEnvironment(earth_angular_velocity_rad_s=0.0)
        sim = AerodynamicBallisticsSimulator(proj_vacuum, env_vacuum)

        v0 = 100.0
        angle = 45.0
        res = sim.simulate(muzzle_velocity_mps=v0, elevation_angle_deg=angle, initial_pos_m=np.array([0.0, 0.0, 0.0]))

        g = env_vacuum.gravity_mps2
        expected_range = (v0 ** 2) / g
        expected_flight_time = 2.0 * v0 * np.sin(np.deg2rad(angle)) / g
        expected_apogee = ((v0 * np.sin(np.deg2rad(angle))) ** 2) / (2.0 * g)

        assert np.isclose(res.maximum_range_m, expected_range, rtol=0.02)
        assert np.isclose(res.flight_time_s, expected_flight_time, rtol=0.02)
        assert np.isclose(res.apogee_altitude_m, expected_apogee, rtol=0.02)

    def test_ballistics_crosswind_drift(self):
        proj = ProjectileProperties()
        # 10 m/s crosswind along East (+X)
        env_wind = BallisticEnvironment(wind_velocity_mps=np.array([10.0, 0.0, 0.0]))
        sim = AerodynamicBallisticsSimulator(proj, env_wind)

        res = sim.simulate(muzzle_velocity_mps=400.0, elevation_angle_deg=20.0, azimuth_angle_deg=0.0)
        # Bullet launched North with East crosswind must drift East (+X)
        assert res.positions_m[-1, 0] > 0.5

    def test_aerodynamic_drag_and_coriolis_deflection(self):
        proj = ProjectileProperties(mass_kg=0.009, cross_section_area_m2=6.36e-5, drag_coefficient_cd=0.30)
        env = BallisticEnvironment(latitude_rad=np.deg2rad(45.0))
        sim = AerodynamicBallisticsSimulator(proj, env)

        res = sim.simulate(muzzle_velocity_mps=800.0, elevation_angle_deg=30.0, azimuth_angle_deg=0.0)
        assert res.maximum_range_m < 20_000.0
        assert res.coriolis_deflection_m > 0.0

    def test_bpa_fluid_area_of_origin_reconstruction(self):
        p_origin_true = np.array([0.0, 1.5, 1.2])
        wall_normal = np.array([0.0, -1.0, 0.0])
        
        stains = []
        impact_points = [
            np.array([-0.8, 3.0, 1.6]),
            np.array([0.7, 3.0, 1.8]),
            np.array([-0.5, 3.0, 0.9]),
            np.array([0.6, 3.0, 0.8]),
        ]

        for idx, p_imp in enumerate(impact_points):
            flight_vec = p_imp - p_origin_true
            dist = np.linalg.norm(flight_vec)
            u_flight = flight_vec / dist
            
            cos_theta = abs(np.dot(u_flight, wall_normal))
            alpha_rad = np.arcsin(np.clip(cos_theta, 0.0, 1.0))
            
            major_l = 8.0
            minor_w = major_l * np.sin(alpha_rad)
            gamma_deg = float(np.rad2deg(np.arctan2(u_flight[2], u_flight[0])))

            stains.append(BloodstainRecord(
                stain_id=idx + 1,
                surface_pos_3d=p_imp,
                major_axis_mm=major_l,
                minor_axis_mm=minor_w,
                gamma_direction_deg=gamma_deg,
                surface_normal=wall_normal
            ))

        res_bpa = BloodstainFluidOriginSolver.solve_origin_3d(stains)
        assert res_bpa.number_of_stains_used == 4
        assert res_bpa.residual_rmse_m < 0.25
        assert np.allclose(res_bpa.origin_point_3d, p_origin_true, atol=0.25)

    def test_bpa_insufficient_stains_rejection(self):
        wall_normal = np.array([0.0, -1.0, 0.0])
        stains = [
            BloodstainRecord(1, np.array([0.0, 3.0, 1.0]), 8.0, 4.0, 0.0, wall_normal),
            BloodstainRecord(2, np.array([1.0, 3.0, 1.0]), 8.0, 4.0, 0.0, wall_normal)
        ]
        with pytest.raises(ValueError, match="Need at least 3 valid bloodstains"):
            BloodstainFluidOriginSolver.solve_origin_3d(stains)


# =====================================================================
# 2. GEO SEISMIC INVERSION (FWI) TESTS
# =====================================================================
class TestGeoSeismicInversion:
    def test_seismic_grid_and_wavelet(self):
        grid = SeismicGridConfig(nx=40, nz=30, dx_m=10.0, dz_m=10.0, dt_s=0.001, nt=150)
        geom = SeismicSurveyGeometry(
            shot_x_indices=[20],
            shot_z_indices=[2],
            receiver_x_indices=[5, 15, 25, 35],
            receiver_z_indices=[2, 2, 2, 2],
            ricker_f0_hz=15.0
        )
        wav = geom.generate_ricker_wavelet(grid.nt, grid.dt_s)
        assert len(wav) == 150
        assert np.isclose(np.max(wav), 1.0, atol=0.05)

    def test_forward_propagation_acoustic_wave(self):
        grid = SeismicGridConfig(nx=40, nz=30, dx_m=10.0, dz_m=10.0, dt_s=0.001, nt=200, nboundary=10)
        geom = SeismicSurveyGeometry(
            shot_x_indices=[20],
            shot_z_indices=[2],
            receiver_x_indices=[10, 20, 30],
            receiver_z_indices=[2, 2, 2]
        )
        fwi = FullWaveformInversion2D(grid, geom)

        v_model = np.full((grid.nz, grid.nx), 2000.0)
        seis, wavefield = fwi.forward_propagate(v_model, shot_idx=0, save_wavefield=True)

        assert seis.shape == (3, 200)
        assert wavefield.shape == (200, grid.total_nz, grid.total_nx)
        assert np.all(np.isfinite(seis))
        assert np.max(np.abs(seis[1, :])) > 0.0

    def test_fwi_adjoint_gradient_and_inversion(self):
        grid = SeismicGridConfig(nx=30, nz=20, dx_m=15.0, dz_m=15.0, dt_s=0.001, nt=150, nboundary=8)
        geom = SeismicSurveyGeometry(
            shot_x_indices=[15],
            shot_z_indices=[2],
            receiver_x_indices=[5, 10, 15, 20, 25],
            receiver_z_indices=[2, 2, 2, 2, 2]
        )
        fwi = FullWaveformInversion2D(grid, geom)

        v_true = np.full((grid.nz, grid.nx), 2000.0)
        v_true[10:15, 12:18] = 2800.0

        v_init = np.full((grid.nz, grid.nx), 2000.0)

        obs_seis, _ = fwi.forward_propagate(v_true, shot_idx=0)

        cost, grad = fwi.compute_adjoint_gradient(v_init, [obs_seis])
        assert cost > 0.0
        assert grad.shape == (grid.nz, grid.nx)
        assert np.any(grad != 0.0)

        res = fwi.invert(v_init, [obs_seis], max_iterations=2, learning_rate_scale=30.0)
        assert isinstance(res, FWIInversionResult)
        assert len(res.cost_history) == 2
        assert res.inverted_velocity_mps.shape == (grid.nz, grid.nx)


# =====================================================================
# 3. INSAR SENTINEL PIPELINE TESTS
# =====================================================================
class TestInSARSentinelPipeline:
    def test_coherence_calculation(self):
        pipeline = SentinelInSARPipeline()
        ny, nx = 40, 40
        s1 = np.random.randn(ny, nx) + 1j * np.random.randn(ny, nx)
        s2 = s1 * np.exp(1j * 0.5)
        
        coh_high = pipeline.compute_coherence(s1, s2, window_size=5)
        assert np.all(coh_high >= 0.0)
        assert np.mean(coh_high) > 0.85

        s3 = np.random.randn(ny, nx) + 1j * np.random.randn(ny, nx)
        coh_low = pipeline.compute_coherence(s1, s3, window_size=5)
        assert np.mean(coh_low) < 0.50

    def test_goldstein_filter_and_phase_unwrapping(self):
        pipeline = SentinelInSARPipeline()
        ny, nx = 32, 32
        
        y_grid, x_grid = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        true_phase = 0.25 * x_grid + 0.15 * y_grid
        wrapped_phase = np.angle(np.exp(1j * true_phase))

        coh = np.ones((ny, nx))
        filt_phase = pipeline.goldstein_filter(wrapped_phase, alpha=0.3, patch_size=16)
        unwrapped = pipeline.unwrap_phase_quality_guided(filt_phase, coh)

        assert unwrapped.shape == (ny, nx)
        d_dx_interior = np.diff(unwrapped, axis=1)[4:-4, 4:-4]
        assert np.isclose(np.mean(d_dx_interior), 0.25, atol=0.08)

    def test_sbas_time_series_inversion_and_landslide_hazard(self):
        pipeline = SentinelInSARPipeline(radar_wavelength_m=SENTINEL1_WAVELENGTH_M)
        
        acqs = [
            SARAcquisition("2026-01-01", 0.0, 0.0),
            SARAcquisition("2026-04-01", 90.0, 25.0),
            SARAcquisition("2026-07-01", 180.0, -15.0),
            SARAcquisition("2026-10-01", 270.0, 30.0)
        ]

        ny, nx = 32, 32
        ifgs = []
        rad_to_mm = - (SENTINEL1_WAVELENGTH_M * 1000.0) / (4.0 * np.pi)
        
        target_disp_90d = -11.08 # mm
        phase_target = target_disp_90d / rad_to_mm

        for i in range(3):
            dt_days = 90.0
            phase_field = np.zeros((ny, nx), dtype=np.float64)
            phase_field[10:22, 10:22] = phase_target
            
            c_ifg = np.exp(1j * phase_field)
            ifgs.append(InterferogramPair(
                pair_id=f"ifg_{i}_{i+1}",
                master_idx=i,
                slave_idx=i+1,
                temporal_baseline_days=dt_days,
                perpendicular_baseline_m=20.0,
                complex_interferogram=c_ifg
            ))

        res = pipeline.invert_sbas_time_series(acqs, ifgs, coherence_threshold=0.2)
        assert isinstance(res, InSARDeformationResult)
        assert res.mean_velocity_mm_per_year.shape == (ny, nx)
        assert res.cumulative_displacement_mm.shape == (4, ny, nx)
        
        assert np.any(res.hazard_alert_mask[12:20, 12:20])
        assert res.max_subsidence_velocity_mm_yr < -35.0
