"""
Comprehensive Unit Test Suite for Advanced Quantum, SCADA & Neuromorphic Systems.

Covers:
1. QKD Satellite Optical Comm & Decoy-State BB84 + CASCADE Protocol.
2. Industrial SCADA DPI Firewall & Stateful Anomaly Detector (Modbus & DNP3).
3. Neuromorphic LIF SNN Event Camera Processor & STDP Target Tracker.
4. Geothermal Borehole Heat Exchanger Transient Simulator & ORC Power Plant.
5. 2D Grad-Shafranov Tokamak MHD Plasma Equilibrium Solver.
"""

import math
import numpy as np
import pytest
import struct
import time

# Import system modules
from qkd_satellite_optical_comm import (
    SatelliteOrbitConfig,
    AtmosphericTurbulenceConfig,
    SPADDetectorConfig,
    DecoyStateBB84Config,
    AtmosphericChannelModel,
    DecoyStateBB84Simulator,
    CascadeErrorReconciliation,
    Basis,
    IntensityState,
)

from scada_modbus_dnp3_firewall import (
    CRC16Calculator,
    ModbusFunctionCode,
    DNP3FunctionCode,
    ModbusSecurityPolicy,
    DNP3SecurityPolicy,
    FirewallAction,
    ModbusTCPFirewall,
    DNP3Firewall,
    SCADAFirewallEngine,
)

from neuromorphic_snn_dvs_tracker import (
    DVSEvent,
    LIFNeuronConfig,
    STDPConfig,
    NeuromorphicPowerConfig,
    DVSStreamGenerator,
    LIFSpikingLayer,
    NeuromorphicTracker,
)

from geothermal_borehole_transient_sim import (
    GroundThermalProperties,
    BoreholeGeometry,
    ORCWorkingFluidConfig,
    BoreholeTransientModel,
    GeothermalORCPowerPlant,
)

from fusion_tokamak_mhd_equilibrium import (
    TokamakGeometry,
    GSGridConfig,
    SolovevEquilibrium,
    GradShafranovSolver,
)


# ============================================================================
# 1. QKD SATELLITE OPTICAL COMM TESTS
# ============================================================================

class TestQKDSatelliteOpticalComm:
    """Test suite for QKD Satellite Optical Communication and CASCADE reconciliation."""

    def test_satellite_orbit_geometry(self):
        orbit = SatelliteOrbitConfig(altitude_km=500.0, zenith_angle_deg=0.0)
        assert pytest.approx(orbit.slant_range_m, rel=1e-3) == 500000.0

        orbit_slant = SatelliteOrbitConfig(altitude_km=500.0, zenith_angle_deg=45.0)
        assert orbit_slant.slant_range_m > 500000.0
        assert orbit_slant.slant_range_m < 800000.0

    def test_atmospheric_channel_model(self):
        orbit = SatelliteOrbitConfig(altitude_km=500.0, zenith_angle_deg=30.0)
        turb = AtmosphericTurbulenceConfig(c_n2=1e-14)
        channel = AtmosphericChannelModel(orbit, turb)

        w_rx = channel.compute_beam_waist_at_receiver()
        assert w_rx > 0.5

        scint_idx = channel.compute_scintillation_index()
        assert 0.0 <= scint_idx <= 1.5

        samples = channel.sample_transmittance(n_samples=50, rng=np.random.default_rng(42))
        assert len(samples) == 50
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)
        assert np.mean(samples) > 0.0

    def test_decoy_bb84_simulation_and_bounds(self):
        sim = DecoyStateBB84Simulator(seed=12345)
        num_pulses = 5000
        alice_bits, alice_bases, alice_intens = sim.generate_alice_pulses(num_pulses)

        assert len(alice_bits) == num_pulses
        assert set(np.unique(alice_bits)).issubset({0, 1})
        assert set(np.unique(alice_bases)).issubset({0, 1})
        assert set(np.unique(alice_intens)).issubset({0, 1, 2})

        bob_bases, bob_clicks, bob_bits = sim.transmit_and_detect(
            alice_bits, alice_bases, alice_intens
        )

        assert len(bob_clicks) == num_pulses
        assert np.sum(bob_clicks) > 0

        sifted = sim.sift_keys(
            alice_bits, alice_bases, alice_intens,
            bob_bases, bob_clicks, bob_bits
        )

        assert 'signal' in sifted and 'decoy' in sifted and 'vacuum' in sifted
        assert sifted['signal']['clicks'] > 0
        assert 0.0 <= sifted['signal']['qber'] < 0.20

        bounds = sim.estimate_decoy_bounds(sifted)
        assert 'Y1_lower' in bounds and 'e1_upper' in bounds
        assert 0.0 < bounds['Y1_lower'] <= 1.0
        assert 0.0 <= bounds['e1_upper'] <= 0.5

        rates = sim.calculate_secret_key_rate(sifted, bounds)
        assert 'secure_key_rate_per_pulse' in rates
        assert rates['secure_key_rate_per_pulse'] >= 0.0

    def test_cascade_error_reconciliation(self):
        rng = np.random.default_rng(999)
        key_len = 512
        alice_key = rng.integers(0, 2, size=key_len, dtype=np.uint8)

        # Introduce 2.5% random bit errors into Bob's key
        bob_key = alice_key.copy()
        error_indices = rng.choice(key_len, size=int(key_len * 0.025), replace=False)
        bob_key[error_indices] = 1 - bob_key[error_indices]

        initial_errors = np.count_nonzero(alice_key != bob_key)
        assert initial_errors > 0

        cascade = CascadeErrorReconciliation(num_iterations=4, target_qber=0.03, seed=42)
        reconciled_bob, parity_bits, final_ber, success = cascade.reconcile(alice_key, bob_key)

        assert success is True
        assert final_ber == 0.0
        assert np.array_equal(alice_key, reconciled_bob)
        assert parity_bits > 0


# ============================================================================
# 2. SCADA MODBUS & DNP3 FIREWALL TESTS
# ============================================================================

class TestSCADAFirewall:
    """Test suite for SCADA DPI Firewall, CRC checks, and MitM anomaly detector."""

    def test_crc16_calculators(self):
        # Modbus RTU CRC test vector: b"123456789" -> 0x4B37
        test_data = b"123456789"
        modbus_crc = CRC16Calculator.calculate_modbus_crc16(test_data)
        assert modbus_crc == 0x4B37

        # DNP3 CRC test vector: standard 8-byte link header
        dnp3_header = bytes([0x05, 0x64, 0x05, 0xC4, 0x00, 0x04, 0x01, 0x00])
        calc_crc = CRC16Calculator.calculate_dnp3_crc16(dnp3_header)
        assert isinstance(calc_crc, int)
        assert 0 <= calc_crc <= 0xFFFF

    def test_modbus_firewall_allowed_and_blocked(self):
        fw = ModbusTCPFirewall()

        # Valid Read Holding Registers (Unit 1, Addr 100, Qty 10)
        valid_packet = struct.pack(">HHHBBHH", 1, 0, 6, 1, 0x03, 100, 10)
        res = fw.inspect_packet(valid_packet)
        assert res.action == FirewallAction.ALLOW

        # Disallowed Unit ID (Unit 99)
        invalid_unit = struct.pack(">HHHBBHH", 2, 0, 6, 99, 0x03, 100, 10)
        res = fw.inspect_packet(invalid_unit)
        assert res.action == FirewallAction.DROP
        assert "Unauthorized Unit ID" in res.reason

        # Disallowed Function Code (0x08 Diagnostics)
        disallowed_fc = struct.pack(">HHHBBHH", 3, 0, 6, 1, 0x08, 0, 0)
        res = fw.inspect_packet(disallowed_fc)
        assert res.action == FirewallAction.DROP
        assert "Disallowed Modbus Function Code" in res.reason

        # Out-of-bounds Register Address Read (Unit 1 allowed 0..1000; request 1500)
        oob_read = struct.pack(">HHHBBHH", 4, 0, 6, 1, 0x03, 1500, 10)
        res = fw.inspect_packet(oob_read)
        assert res.action == FirewallAction.DROP
        assert "out of bounds" in res.reason

        # Value Limit Alert (Unit 1 allowed 0..10000; write single register 50000)
        oob_write_val = struct.pack(">HHHBBHH", 5, 0, 6, 1, 0x06, 50, 50000)
        res = fw.inspect_packet(oob_write_val)
        assert res.action == FirewallAction.ALERT
        assert "exceeds safety limits" in res.reason

    def test_modbus_slew_rate_anomaly(self):
        policy = ModbusSecurityPolicy(max_write_slew_rate=50.0)
        fw = ModbusTCPFirewall(policy)

        pkt1 = struct.pack(">HHHBBHH", 1, 0, 6, 1, 0x06, 20, 100)
        res1 = fw.inspect_packet(pkt1)
        assert res1.action == FirewallAction.ALLOW

        pkt2 = struct.pack(">HHHBBHH", 2, 0, 6, 1, 0x06, 20, 1000)
        res2 = fw.inspect_packet(pkt2)
        assert res2.action == FirewallAction.ALERT
        assert "slew-rate anomaly" in res2.reason

    def test_dnp3_firewall_and_sbo_mitm_mitigation(self):
        fw = DNP3Firewall()

        def build_dnp3_packet(src: int, dst: int, func: int, seq: int, obj_data: bytes = b"") -> bytes:
            payload = bytearray([0xC0, 0xC0 | (seq & 0x0F), func]) + bytearray(obj_data)
            length = 5 + len(payload)
            hdr = bytearray([0x05, 0x64, length, 0xC4]) + bytearray(struct.pack("<HH", dst, src))
            hdr_crc = CRC16Calculator.calculate_dnp3_crc16(hdr)
            hdr += struct.pack("<H", hdr_crc)

            pay_crc = CRC16Calculator.calculate_dnp3_crc16(payload)
            return bytes(hdr + payload + struct.pack("<H", pay_crc))

        # 1. Valid Link Layer Packet
        valid_select = build_dnp3_packet(src=1, dst=1024, func=DNP3FunctionCode.SELECT.value, seq=1, obj_data=b"\x01\x02")
        res = fw.inspect_packet(valid_select)
        assert res.action == FirewallAction.ALLOW
        assert "SELECT registered" in res.reason

        # 2. Matching Valid OPERATE Packet (SBO sequence)
        valid_operate = build_dnp3_packet(src=1, dst=1024, func=DNP3FunctionCode.OPERATE.value, seq=1, obj_data=b"\x01\x02")
        res = fw.inspect_packet(valid_operate)
        assert res.action == FirewallAction.ALLOW

        # 3. MitM Anomaly: OPERATE without prior SELECT
        unsolicited_operate = build_dnp3_packet(src=1, dst=1024, func=DNP3FunctionCode.OPERATE.value, seq=7, obj_data=b"\x01\x02")
        res = fw.inspect_packet(unsolicited_operate)
        assert res.action == FirewallAction.ALERT
        assert "without prior SELECT" in res.reason

        # 4. Blocked Restart Command (0x0D COLD_RESTART)
        restart_pkt = build_dnp3_packet(src=1, dst=1024, func=DNP3FunctionCode.COLD_RESTART.value, seq=2)
        res = fw.inspect_packet(restart_pkt)
        assert res.action == FirewallAction.DROP
        assert "Restart command blocked" in res.reason

        # 5. Tampered CRC detection
        tampered_pkt = bytearray(valid_select)
        tampered_pkt[12] ^= 0xFF
        res = fw.inspect_packet(bytes(tampered_pkt))
        assert res.action == FirewallAction.DROP
        assert "CRC failure" in res.reason


# ============================================================================
# 3. NEUROMORPHIC SNN & DVS TRACKER TESTS
# ============================================================================

class TestNeuromorphicSNNTracker:
    """Test suite for LIF Spiking Neural Network, STDP, and microsecond DVS tracking."""

    def test_lif_neuron_dynamics_and_spiking(self):
        layer = LIFSpikingLayer(width=10, height=10)
        cfg = layer.config

        assert layer.v_membrane[5, 5] == cfg.v_rest

        spiked = layer.inject_current_and_update(5, 5, current_t_us=1000, weight=0.2)
        assert spiked is False
        assert layer.v_membrane[5, 5] > cfg.v_rest

        spiked = layer.inject_current_and_update(5, 5, current_t_us=2000, weight=5.0)
        assert spiked is True
        assert layer.v_membrane[5, 5] == cfg.v_reset
        assert layer.spike_count == 1

        spiked_during_refractory = layer.inject_current_and_update(5, 5, current_t_us=2500, weight=10.0)
        assert spiked_during_refractory is False

    def test_stdp_learning(self):
        tracker = NeuromorphicTracker(width=16, height=16)
        w_init = float(tracker.weights[5, 5, 1, 1])

        tracker.apply_stdp(5, 5, pre_spike_t_us=10000, post_spike_t_us=15000)
        w_potentiated = float(tracker.weights[5, 5, 1, 1])
        assert w_potentiated > w_init

        tracker.apply_stdp(5, 5, pre_spike_t_us=20000, post_spike_t_us=15000)
        w_depressed = float(tracker.weights[5, 5, 1, 1])
        assert w_depressed < w_potentiated

    def test_dvs_stream_generation_and_power_benchmark(self):
        events = DVSStreamGenerator.generate_moving_target(
            width=32, height=32, duration_us=50000,
            velocity_px_per_s=(100.0, 80.0), seed=42
        )
        assert len(events) > 50
        assert events[0].timestamp_us <= events[-1].timestamp_us

        tracker = NeuromorphicTracker(width=32, height=32)
        results = tracker.process_event_stream(events)

        assert results['processed_events'] == len(events)
        assert results['total_sops'] > 0
        assert results['power_under_10mw'] is True
        assert results['total_power_mw'] < 10.0
        assert results['total_power_mw'] >= results['static_power_mw']


# ============================================================================
# 4. GEOTHERMAL BOREHOLE & ORC POWER TESTS
# ============================================================================

class TestGeothermalORCSimulator:
    """Test suite for GHE transient thermal response & Organic Rankine Cycle."""

    def test_borehole_thermal_resistance(self):
        ground = GroundThermalProperties()
        geo = BoreholeGeometry()
        model = BoreholeTransientModel(ground, geo)

        r_b = model.calculate_borehole_thermal_resistance()
        assert 0.04 <= r_b <= 0.25

    def test_ils_transient_temperature_response(self):
        ground = GroundThermalProperties(thermal_conductivity_w_mk=2.8)
        geo = BoreholeGeometry()
        model = BoreholeTransientModel(ground, geo)

        delta_t = model.ils_temperature_response(time_seconds=360000.0, heat_rate_w_m=50.0)
        assert delta_t > 0.0
        assert delta_t < 25.0

    def test_multi_step_load_superposition(self):
        ground = GroundThermalProperties()
        geo = BoreholeGeometry()
        model = BoreholeTransientModel(ground, geo)

        time_hours = np.array([10.0, 20.0, 50.0, 100.0])
        heat_loads = np.array([40.0, 50.0, -30.0, 20.0])

        fluid_temps = model.simulate_transient_load_profile(time_hours, heat_loads)
        assert len(fluid_temps) == len(time_hours)
        assert np.all(np.isfinite(fluid_temps))
        assert fluid_temps[1] > ground.undisturbed_temp_c

    def test_trt_curve_fitting(self):
        ground = GroundThermalProperties(thermal_conductivity_w_mk=2.6)
        geo = BoreholeGeometry()
        model = BoreholeTransientModel(ground, geo)

        t_hours = np.linspace(10.0, 70.0, 50)
        q_const = 50.0
        synthetic_temps = model.simulate_transient_load_profile(t_hours, np.full_like(t_hours, q_const))

        k_s_fit, r_b_fit = model.perform_trt_curve_fit(t_hours, synthetic_temps, q_const)
        assert pytest.approx(k_s_fit, rel=0.15) == ground.thermal_conductivity_w_mk
        assert r_b_fit > 0.0

    def test_orc_power_cycle_performance(self):
        plant = GeothermalORCPowerPlant()
        perf = plant.calculate_cycle_performance(
            brine_inlet_temp_c=145.0,
            brine_mass_flow_kg_s=30.0,
            condenser_temp_c=35.0
        )

        assert perf['net_electric_power_kw'] > 0.0
        assert perf['net_electric_power_mw'] > 0.3
        assert 0.05 <= perf['thermal_efficiency'] <= 0.25
        assert 0.15 <= perf['carnot_efficiency'] <= 0.40
        assert 0.10 <= perf['exergy_efficiency'] <= 0.70


# ============================================================================
# 5. FUSION TOKAMAK MHD EQUILIBRIUM TESTS
# ============================================================================

class TestTokamakMHDEquilibrium:
    """Test suite for 2D Grad-Shafranov PDE equilibrium solver and Solov'ev profiles."""

    def test_solovev_analytical_equilibrium(self):
        geo = TokamakGeometry(major_radius_m=6.2, elongation_kappa=1.7)
        sol = SolovevEquilibrium(geo, q_0=1.1)

        psi_axis = sol.evaluate_psi(np.array([6.2]), np.array([0.0]))[0]
        assert pytest.approx(psi_axis, abs=1e-5) == 0.0

        psi_off = sol.evaluate_psi(np.array([7.0]), np.array([1.0]))[0]
        assert psi_off > 0.0

        b_r, b_z, b_phi, b_tot = sol.evaluate_b_fields(np.array([6.2]), np.array([0.0]))
        assert pytest.approx(b_r[0], abs=1e-4) == 0.0
        assert pytest.approx(b_z[0], abs=1e-4) == 0.0
        assert pytest.approx(b_phi[0], rel=1e-3) == geo.b_toroidal_0_t

    def test_grad_shafranov_numerical_solver(self):
        geo = TokamakGeometry(major_radius_m=6.2, b_toroidal_0_t=5.3, total_plasma_current_ma=15.0)
        grid = GSGridConfig(nr=33, nz=33, max_iterations=150, convergence_tol=1e-4)
        solver = GradShafranovSolver(geo, grid)

        solver.initialize_with_solovev()
        diag = solver.solve()

        assert diag['iterations'] > 0
        assert 5.0 <= diag['magnetic_axis_R_m'] <= 7.0
        assert abs(diag['magnetic_axis_Z_m']) < 0.5
        assert diag['integrated_plasma_current_A'] > 0.0

        b_r, b_z, b_phi, b_tot = solver.compute_b_fields()
        assert b_r.shape == (grid.nz, grid.nr)
        assert np.all(np.isfinite(b_tot))
        assert np.all(b_tot > 0.0)

    def test_safety_factor_profile(self):
        geo = TokamakGeometry()
        grid = GSGridConfig(nr=25, nz=25)
        solver = GradShafranovSolver(geo, grid)
        solver.initialize_with_solovev()

        norm_levels, q_profile = solver.compute_safety_factor_profile(n_surfaces=8)
        assert len(q_profile) == 8
        assert np.all(q_profile >= 1.0)
        assert q_profile[-1] > q_profile[0]
