"""
Extended OPFOR & Comprehensive Stress Test Suite for:
1. acoustic_westervelt_parametric_array.py
2. seismic_lithospheric_rate_state_friction.py
3. phononic_subsea_glider_tinyml.py
4. maxwell_lienard_wiechert_directed_energy.py
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from acoustic_westervelt_parametric_array import (
    AirMediumProperties,
    ParametricTransducerConfig,
    WesterveltParametricSolver,
    WesterveltSimulationResult
)

from seismic_lithospheric_rate_state_friction import (
    FaultMaterialParameters,
    SpringSliderConfig,
    StateEvolutionLaw,
    LithosphericRateStateSimulator,
    SeismicEvent,
    SeismogenesisResult
)

from phononic_subsea_glider_tinyml import (
    PhononicLayerMaterial,
    PhononicCrystalUnitCell,
    PhononicMetamaterialSolver,
    TinyMLAcousticClassifier,
    QuantizedLayerINT8
)

from maxwell_lienard_wiechert_directed_energy import (
    RelativisticLienardWiechertSolver,
    C_VACUUM,
    EPSILON_0,
    MU_0,
    ELEMENTARY_CHARGE
)


# =====================================================================
# 1. ACOUSTIC WESTERVELT & PARAMETRIC ARRAY TESTS
# =====================================================================
class TestAcousticWestervelt:
    def test_air_medium_properties(self):
        air = AirMediumProperties()
        assert air.c0 == 343.2
        assert air.rho0 == 1.204
        assert np.isclose(air.acoustic_impedance, 1.204 * 343.2)
        
        # Absorption scales quadratically with frequency
        alpha_10k = air.absorption_coefficient(10_000.0)
        alpha_100k = air.absorption_coefficient(100_000.0)
        assert np.isclose(alpha_100k / alpha_10k, 100.0, rtol=1e-3)

    def test_transducer_config_properties(self):
        cfg = ParametricTransducerConfig(f1_hz=100_000.0, f2_hz=102_000.0, source_radius_m=0.04)
        assert cfg.difference_freq_hz == 2000.0
        assert cfg.mean_carrier_freq_hz == 101_000.0
        assert np.isclose(cfg.source_area_m2, np.pi * (0.04 ** 2))
        assert cfg.rayleigh_distance_m > 0.0

    def test_berktay_demodulation_analytical(self):
        solver = WesterveltParametricSolver()
        z_targets = np.array([0.5, 1.0, 2.0])
        res = solver.solve_berktay_demodulation(z_distances=z_targets, t_duration_s=0.005)
        
        assert "p_audio_matrix" in res
        assert "audio_spl_db" in res
        assert len(res["audio_spl_db"]) == 3
        # In far-field, pressure decays as 1/z, so SPL decreases with distance
        assert res["audio_spl_db"][0] > res["audio_spl_db"][2]

    def test_progressive_westervelt_solver(self):
        cfg = ParametricTransducerConfig(f1_hz=100_000.0, f2_hz=101_000.0, p0_peak_pa=400.0)
        solver = WesterveltParametricSolver(transducer=cfg)
        sim_res = solver.solve_progressive_westervelt_1d(z_max_m=0.5, nz=50, t_periods=20, samples_per_period=16)

        assert isinstance(sim_res, WesterveltSimulationResult)
        assert sim_res.p_matrix_pa.shape[0] == 50
        assert len(sim_res.audio_spl_db) == 50
        assert np.all(np.isfinite(sim_res.p_matrix_pa))
        
        # Audio beamwidth calculation
        bw = solver.calculate_parametric_beamwidth_deg(sim_res)
        assert 0.0 < bw < 45.0  # Sharp directional pencil beam characteristics

    def test_westervelt_beamwidth_empty_edge_cases(self):
        solver = WesterveltParametricSolver()
        dummy_res = WesterveltSimulationResult(
            z_grid_m=np.array([0.0]),
            t_grid_s=np.array([0.0]),
            p_matrix_pa=np.zeros((1, 1)),
            demodulated_audio_pa=np.zeros(1),
            audio_spl_db=np.zeros(1),
            carrier_spl_db=np.zeros(1),
            spectral_frequencies_hz=np.zeros(1),
            power_spectral_density=np.zeros(1),
            beam_angles_rad=None,
            beam_directivity_db=None
        )
        assert solver.calculate_parametric_beamwidth_deg(dummy_res) == 0.0


# =====================================================================
# 2. SEISMIC RATE-AND-STATE SEISMOGENESIS TESTS
# =====================================================================
class TestSeismicRateAndState:
    def test_fault_material_mechanics(self):
        mat = FaultMaterialParameters(a=0.008, b=0.012, sigma_n_pa=50e6, dc_m=1e-4)
        assert mat.is_velocity_weakening is True
        assert mat.critical_stiffness_k_crit > 0.0
        assert mat.radiation_damping_eta > 0.0
        assert np.isfinite(mat.critical_nucleation_length_m)

    def test_friction_and_state_evolution(self):
        mat = FaultMaterialParameters()
        cfg_aging = SpringSliderConfig(state_law=StateEvolutionLaw.AGING_LAW)
        sim_aging = LithosphericRateStateSimulator(mat, cfg_aging)
        
        mu_steady = sim_aging.friction_coefficient(v=1e-6, theta=1e-4/1e-6)
        assert np.isclose(mu_steady, mat.mu0, atol=1e-4)
        
        # Steady state state-derivative should be ~0
        dth_dt = sim_aging.state_derivative(v=1e-6, theta=1e-4/1e-6)
        assert np.isclose(dth_dt, 0.0, atol=1e-6)

        # Ruina slip law
        cfg_slip = SpringSliderConfig(state_law=StateEvolutionLaw.SLIP_LAW)
        sim_slip = LithosphericRateStateSimulator(mat, cfg_slip)
        dth_slip = sim_slip.state_derivative(v=1e-6, theta=1e-4/1e-6)
        assert np.isclose(dth_slip, 0.0, atol=1e-6)

    def test_stick_slip_earthquake_cycle_simulation(self):
        # Velocity weakening setup with K < k_crit to generate dynamic events
        mat = FaultMaterialParameters(
            sigma_n_pa=20.0e6,
            a=0.005,
            b=0.015, # Strong velocity weakening (b - a = 0.010)
            dc_m=1.0e-4,
            shear_modulus_pa=30.0e9
        )
        cfg = SpringSliderConfig(
            v_load_mps=1.0e-3, # Accelerated loading for rapid cycle integration
            stiffness_k_pa_per_m=0.5e9, # K < k_crit => unstable stick-slip
            fault_area_m2=1.0e6,
            v_init_mps=1.0e-4, # Initial perturbation off steady state
            theta_init_s=1.0e-4 / 1.0e-4
        )
        sim = LithosphericRateStateSimulator(mat, cfg)
        res = sim.simulate(t_max_s=2.0, rtol=1e-5, atol=1e-7)

        assert isinstance(res, SeismogenesisResult)
        assert len(res.time_s) > 10
        assert len(res.events) >= 1
        
        ev0 = res.events[0]
        assert ev0.peak_velocity_mps > 1e-3
        assert ev0.coseismic_slip_m > 0.0
        assert ev0.seismic_moment_nm > 0.0
        assert ev0.moment_magnitude_mw > -5.0
        assert ev0.radiated_energy_joules >= 0.0
        assert ev0.fracture_energy_joules >= 0.0

    def test_slip_law_ode_simulation(self):
        mat = FaultMaterialParameters(
            sigma_n_pa=20.0e6,
            a=0.005,
            b=0.015,
            dc_m=1.0e-4,
            shear_modulus_pa=30.0e9
        )
        cfg_slip = SpringSliderConfig(
            state_law=StateEvolutionLaw.SLIP_LAW,
            v_load_mps=1.0e-3,
            stiffness_k_pa_per_m=0.5e9,
            v_init_mps=1.0e-4,
            theta_init_s=1.0e-4 / 1.0e-4
        )
        sim = LithosphericRateStateSimulator(mat, cfg_slip)
        res = sim.simulate(t_max_s=2.0, rtol=1e-5, atol=1e-7)
        assert isinstance(res, SeismogenesisResult)
        assert len(res.events) >= 1

    def test_velocity_strengthening_stability(self):
        # Velocity strengthening (a > b): absolutely stable, NO earthquakes
        mat = FaultMaterialParameters(a=0.015, b=0.005)
        assert mat.critical_nucleation_length_m == float('inf')
        cfg = SpringSliderConfig(v_load_mps=1e-4, stiffness_k_pa_per_m=5e9, v_init_mps=1e-4)
        sim = LithosphericRateStateSimulator(mat, cfg)
        res = sim.simulate(t_max_s=2.0)
        
        assert len(res.events) == 0
        assert np.max(res.velocity_mps) < 1e-2


# =====================================================================
# 3. PHONONIC METAMATERIAL & TINYML CLASSIFIER TESTS
# =====================================================================
class TestPhononicTinyML:
    def test_phononic_dispersion_bandgaps(self):
        layer_a = PhononicLayerMaterial(name="Polyurethane", density_kg_m3=1100.0, sound_speed_mps=1500.0, thickness_m=0.02)
        layer_b = PhononicLayerMaterial(name="Steel", density_kg_m3=7800.0, sound_speed_mps=5900.0, thickness_m=0.02)
        unit_cell = PhononicCrystalUnitCell(layer_a=layer_a, layer_b=layer_b, num_periods=6)
        
        assert unit_cell.lattice_constant_m == 0.04
        assert unit_cell.total_thickness_m == 0.24
        assert unit_cell.bragg_frequency_hz > 0.0

        solver = PhononicMetamaterialSolver(unit_cell)
        disp_res = solver.compute_dispersion(freq_min_hz=500.0, freq_max_hz=25_000.0, num_freqs=100)

        assert len(disp_res.frequencies_hz) == 100
        assert np.all(disp_res.transmission_db <= 0.0)
        assert len(disp_res.bandgaps_hz) >= 1
        min_trans = np.min(disp_res.transmission_db)
        assert min_trans < -10.0

    def test_transfer_matrix_unitarity_determinant(self):
        mat = PhononicLayerMaterial(name="Water", density_kg_m3=1000.0, sound_speed_mps=1500.0, thickness_m=0.01)
        cell = PhononicCrystalUnitCell(layer_a=mat, layer_b=mat)
        solver = PhononicMetamaterialSolver(cell)
        m = solver.compute_layer_matrix(mat, freq_hz=5000.0)
        det_m = np.linalg.det(m)
        assert np.isclose(det_m.real, 1.0, atol=1e-6)
        assert np.isclose(det_m.imag, 0.0, atol=1e-6)

    def test_tinyml_classifier_quantization_and_inference(self):
        classifier = TinyMLAcousticClassifier(random_seed=123)
        assert classifier.layer1.weights_int8.dtype == np.int8
        assert classifier.layer1.biases_int32.dtype == np.int32
        
        t = np.linspace(0, 0.1, 800)
        sub_audio = np.sin(2 * np.pi * 150 * t) + 0.3 * np.random.randn(len(t))
        
        # Test feature extraction with and without metamaterial filter
        layer_a = PhononicLayerMaterial("Poly", 1100.0, 1500.0, 0.02)
        layer_b = PhononicLayerMaterial("Steel", 7800.0, 5900.0, 0.02)
        cell = PhononicCrystalUnitCell(layer_a, layer_b, num_periods=4)
        solver = PhononicMetamaterialSolver(cell)
        disp_filter = solver.compute_dispersion(freq_min_hz=100.0, freq_max_hz=4000.0, num_freqs=50)

        features_raw = classifier.extract_features(sub_audio, sample_rate_hz=8000.0)
        features_filtered = classifier.extract_features(sub_audio, sample_rate_hz=8000.0, metamaterial_filter=disp_filter)
        
        assert features_raw.shape == (16,)
        assert features_filtered.shape == (16,)
        
        # Quantize to INT8
        q_feat = classifier.quantize_input(features_raw)
        assert q_feat.dtype == np.int8
        assert len(q_feat) == 16
        
        # Perform INT8 inference
        pred = classifier.predict_int8(q_feat)
        assert "predicted_class_id" in pred
        assert 0 <= pred["predicted_class_id"] < 4
        assert np.isclose(np.sum(pred["probabilities"]), 1.0, atol=1e-5)
        
        # Ensure memory footprint fits in ultra-low power subsea glider MCU (< 32 KB SRAM)
        assert classifier.get_memory_footprint_bytes() < 32768


# =====================================================================
# 4. MAXWELL LIENARD-WIECHERT DIRECTED ENERGY TESTS
# =====================================================================
class TestMaxwellLienardWiechert:
    def test_retarded_time_and_potentials_static_limit(self):
        solver = RelativisticLienardWiechertSolver(charge_coulombs=ELEMENTARY_CHARGE)
        
        def static_traj(t: float):
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

        r_obs = np.array([1.0, 0.0, 0.0])
        t_obs = 10.0
        
        res = solver.evaluate_fields(t_obs, r_obs, static_traj)
        
        expected_t_ret = t_obs - (1.0 / C_VACUUM)
        assert np.isclose(res.t_ret_s, expected_t_ret, atol=1e-12)
        
        expected_phi = ELEMENTARY_CHARGE / (4.0 * np.pi * EPSILON_0 * 1.0)
        assert np.isclose(res.phi_potential_volts, expected_phi, rtol=1e-4)
        
        assert np.allclose(res.e_radiation_field_v_per_m, [0.0, 0.0, 0.0], atol=1e-20)
        assert np.isclose(res.total_instantaneous_radiated_power_w, 0.0, atol=1e-25)

    def test_synchrotron_radiation_beaming(self):
        solver = RelativisticLienardWiechertSolver(charge_coulombs=ELEMENTARY_CHARGE)
        traj = solver.synchrotron_trajectory(radius_m=5.0, energy_gev=1.0)
        
        t_obs = 0.0
        r_obs = np.array([10.0, 0.0, 0.0])
        
        res = solver.evaluate_fields(t_obs, r_obs, traj)
        
        assert np.linalg.norm(res.e_radiation_field_v_per_m) > 0.0
        assert res.poynting_flux_magnitude > 0.0
        assert res.total_instantaneous_radiated_power_w > 0.0
        assert res.radiated_power_per_solid_angle_w_sr >= 0.0

    def test_undulator_fel_directed_energy(self):
        solver = RelativisticLienardWiechertSolver(charge_coulombs=ELEMENTARY_CHARGE)
        traj = solver.undulator_fel_trajectory(undulator_period_m=0.02, k_parameter=1.2, gamma=500.0)
        
        t_obs = 1e-8
        r_obs = np.array([0.1, 0.0, 5.0])
        
        res = solver.evaluate_fields(t_obs, r_obs, traj)
        assert np.isfinite(res.phi_potential_volts)
        assert np.all(np.isfinite(res.e_total_field_v_per_m))
        assert np.all(np.isfinite(res.b_total_field_tesla))
        assert np.all(np.isfinite(res.poynting_vector_w_per_m2))
        assert res.total_instantaneous_radiated_power_w > 0.0

    def test_bremsstrahlung_radiation(self):
        solver = RelativisticLienardWiechertSolver(charge_coulombs=ELEMENTARY_CHARGE)
        traj = solver.linear_bremsstrahlung_trajectory(v0_mps=0.8 * C_VACUUM, deceleration_mps2=1e16, t_stop_s=1e-8)
        
        t_obs = 1e-9
        r_obs = np.array([2.0, 2.0, 0.0])
        
        res = solver.evaluate_fields(t_obs, r_obs, traj)
        assert res.radiated_power_per_solid_angle_w_sr >= 0.0
        assert np.isfinite(res.poynting_flux_magnitude)
