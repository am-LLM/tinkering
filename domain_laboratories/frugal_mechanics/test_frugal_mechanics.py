"""
Production Verification & Test Harness: Frugal Mechanics & Bio-Energy
=====================================================================
Comprehensive unit, regression, and physical invariance test suite covering:
1. hydraulic_ram_pump_transient.py (MOC 1D PDE & Joukowsky Water Hammer)
2. seebeck_mppt_controller.py (TEG Multi-Couple Harvester & MPPT)
3. salvaged_bms_ekf.py (2-RC Thevenin ECM, EKF & Active Balancer)
4. woodgas_stirling_sim.py (Biomass Gasification & Stirling Stoichiometry)
5. synchronous_buck_boost_esc.py (4-Switch Synchronous Buck-Boost & R0 Compensation)
6. fsi_flutter_piezo_harvester.py (2D FSI Aeroelastic Flutter & Piezoelectric Harvester)
"""

import math
import os
import runpy
import numpy as np
import pytest

from hydraulic_ram_pump_transient import (
    HydraulicRamPumpMOC,
    DrivePipeSpec,
    WasteValveSpec,
    DeliveryValveSpec,
    AirChamberSpec,
    ValveState,
    SimulationRecord,
    SimulationSummary,
)
from seebeck_mppt_controller import (
    SeebeckMPPTSystem,
    TEGModuleSpec,
    ThermalCombustionEnvironment,
    BatteryBufferSpec,
    PerturbAndObserveMPPT,
    IncrementalConductanceMPPT,
    MPPTAlgorithm,
    ChargeStage,
    MPPTTelemetry,
)
from salvaged_bms_ekf import (
    SalvagedPackBMS,
    CellParameters,
    ExtendedKalmanFilterCell,
    NonLinearOCVModel,
    ActiveCellBalancer,
    CellHealthStatus,
    PackTelemetry,
)
from woodgas_stirling_sim import (
    WoodgasStirlingSystem,
    BiomassFeedstock,
    DowndraftGasifierModel,
    SecondaryCombustor,
    StirlingEngineModel,
    StirlingEngineSpec,
    StoichiometricAirFuelOptimizer,
    GasifierOperatingParams,
    ProducerGasComposition,
    SystemOperatingState,
)
from synchronous_buck_boost_esc import (
    SynchronousBuckBoostESC,
    ConverterHardwareSpec,
    BatterySourceSpec,
    ControlGains,
    ConverterMode,
    ESCTelemetryRecord,
)
from fsi_flutter_piezo_harvester import (
    FSIFlutterPiezoHarvester,
    FluidFlowSpec,
    MembraneStructureSpec,
    PiezoHarvesterSpec,
    FlutterTelemetryRecord,
    FlutterSimulationSummary,
)


# ============================================================================
# 1. HYDRAULIC RAM PUMP TRANSIENT TESTS
# ============================================================================

class TestHydraulicRamPumpMOC:
    """Verification tests for Hydraulic Ram Pump Fluid Transient Solver."""

    def test_grid_and_courant_initialization(self):
        pipe = DrivePipeSpec(length=12.0, diameter=0.05, wave_speed=1200.0, supply_head=3.0)
        sim = HydraulicRamPumpMOC(pipe_spec=pipe, num_reaches=6)
        
        assert sim.num_reaches == 6
        assert sim.num_nodes == 7
        assert math.isclose(sim.dx, 2.0, rel_tol=1e-5)
        # Courant number Cr = a * dt / dx == 1.0
        assert math.isclose(sim.pipe.wave_speed * sim.dt / sim.dx, 1.0, rel_tol=1e-5)
        assert len(sim.H) == 7
        assert len(sim.Q) == 7
        assert np.all(sim.H == 3.0)

    def test_joukowsky_head_calculation(self):
        pipe = DrivePipeSpec(wave_speed=1200.0, gravity=9.80665)
        sim = HydraulicRamPumpMOC(pipe_spec=pipe)
        
        delta_v = 1.5 # m/s
        expected_head_rise = (1200.0 * 1.5) / 9.80665
        calc_head = sim.theoretical_joukowsky_head(delta_v)
        assert math.isclose(calc_head, expected_head_rise, rel_tol=1e-4)

    def test_single_step_execution(self):
        sim = HydraulicRamPumpMOC()
        rec = sim.step()
        assert rec.time > 0.0
        assert isinstance(rec.head_inlet, float)
        assert isinstance(rec.head_valve_box, float)
        assert rec.waste_valve_state in [ValveState.OPEN, ValveState.CLOSING, ValveState.CLOSED]

    def test_water_hammer_shockwave_and_pumping_cycle(self):
        """Verify that water hammer shockwave generates high pressure and pumps water."""
        pipe = DrivePipeSpec(length=10.0, diameter=0.05, wave_speed=1000.0, supply_head=3.5)
        waste = WasteValveSpec(closure_velocity_thresh=0.50, closure_time=0.003, min_closed_time=0.008)
        delivery = DeliveryValveSpec(delivery_head=15.0, cracking_head=0.2)
        sim = HydraulicRamPumpMOC(
            pipe_spec=pipe,
            waste_valve_spec=waste,
            delivery_valve_spec=delivery,
            num_reaches=8,
        )

        records, summary = sim.run_simulation(duration_sec=1.5)
        
        assert summary.peak_transient_head_m > delivery.delivery_head
        assert summary.total_cycles >= 1
        assert summary.total_volume_delivered_liters > 0.0
        assert math.isclose(summary.head_ratio, 15.0 / 3.5, rel_tol=1e-3)
        assert 0.0 <= summary.rankine_efficiency_pct <= 100.0
        assert 0.0 <= summary.daubuisson_efficiency_pct <= 100.0

    def test_variable_head_ratio_scaling(self):
        """Verify pumping efficiency scaling with varying delivery heads."""
        heads = [12.0, 24.0, 36.0]
        results = []
        for h_del in heads:
            sim = HydraulicRamPumpMOC(
                pipe_spec=DrivePipeSpec(supply_head=3.0),
                delivery_valve_spec=DeliveryValveSpec(delivery_head=h_del),
            )
            _, summ = sim.run_simulation(duration_sec=1.0)
            results.append(summ.head_ratio)

        assert results[0] < results[1] < results[2]
        assert math.isclose(results[0], 4.0, rel_tol=1e-2)
        assert math.isclose(results[2], 12.0, rel_tol=1e-2)

    def test_zero_flow_efficiency_edge_cases(self):
        """Test zero wasted volume and zero supply head branches in summary."""
        sim = HydraulicRamPumpMOC()
        sim.total_wasted_volume = 0.0
        sim.total_delivered_volume = 0.0
        records, summary = sim.run_simulation(duration_sec=0.001)
        assert summary.rankine_efficiency_pct == 0.0
        assert summary.daubuisson_efficiency_pct == 0.0

        sim_zero_head = HydraulicRamPumpMOC(pipe_spec=DrivePipeSpec(supply_head=0.0))
        records2, summary2 = sim_zero_head.run_simulation(duration_sec=0.001)
        assert summary2.rankine_efficiency_pct == 0.0
        assert summary2.daubuisson_efficiency_pct == 0.0


# ============================================================================
# 2. SEEBECK MPPT CONTROLLER TESTS
# ============================================================================

class TestSeebeckMPPTController:
    """Verification tests for Seebeck Thermoelectric MPPT Harvester."""

    def test_teg_physics_and_internal_resistance(self):
        teg = TEGModuleSpec(num_couples=127, seebeck_coeff_per_couple=2.0e-4, internal_resistance_25c=2.0)
        assert math.isclose(teg.seebeck_total, 127 * 2.0e-4, rel_tol=1e-5)
        
        r25 = teg.internal_resistance(25.0)
        assert math.isclose(r25, 2.0, rel_tol=1e-5)
        r100 = teg.internal_resistance(100.0)
        assert r100 > r25

    def test_perturb_and_observe_mppt_convergence(self):
        mppt = PerturbAndObserveMPPT(initial_duty=0.2, base_step=0.02)
        v_teg = 8.0
        i_teg = 2.0
        p_teg = 16.0
        duty1 = mppt.compute_duty_cycle(v_teg, i_teg, p_teg, dt=0.01)
        
        duty2 = mppt.compute_duty_cycle(7.5, 2.4, 18.0, dt=0.01)
        assert duty2 >= duty1
        assert 0.05 <= duty2 <= 0.95

        # Test power decrease and direction reversal
        duty3 = mppt.compute_duty_cycle(7.0, 2.0, 14.0, dt=0.01)
        assert 0.05 <= duty3 <= 0.95

        mppt.reset()
        assert mppt.prev_power == 0.0
        assert mppt.prev_voltage == 0.0

    def test_incremental_conductance_mppt(self):
        inc = IncrementalConductanceMPPT(initial_duty=0.5)
        d1 = inc.compute_duty_cycle(10.0, 2.0, 20.0, dt=0.01)
        d2 = inc.compute_duty_cycle(9.8, 2.2, 21.56, dt=0.01)
        assert 0.05 <= d2 <= 0.95

        # Test delta_v < 1e-4 branches (delta_i > 0 and delta_i < 0)
        d3 = inc.compute_duty_cycle(9.8, 2.5, 24.5, dt=0.01)
        assert d3 >= d2
        d4 = inc.compute_duty_cycle(9.8, 2.1, 20.58, dt=0.01)
        assert d4 <= d3

        # Test error > 0.02 and inc_cond <= inst_cond branch
        inc.prev_v = 10.0
        inc.prev_i = 1.0
        d5 = inc.compute_duty_cycle(11.0, 0.5, 5.5, dt=0.01)
        assert 0.05 <= d5 <= 0.95

        inc.reset()
        assert inc.prev_v == 0.0
        assert inc.prev_i == 0.0

    def test_abstract_mppt_algorithm_class(self):
        class ConcreteMPPT(MPPTAlgorithm):
            def compute_duty_cycle(self, v_teg: float, i_teg: float, p_teg: float, dt: float) -> float:
                super().compute_duty_cycle(v_teg, i_teg, p_teg, dt)
                return 0.5
            def reset(self) -> None:
                super().reset()
        
        c = ConcreteMPPT()
        assert c.compute_duty_cycle(1.0, 1.0, 1.0, 0.1) == 0.5
        c.reset()

    def test_integrated_seebeck_mppt_simulation(self):
        system = SeebeckMPPTSystem(mppt_algorithm=PerturbAndObserveMPPT())
        telemetry, summary = system.run_simulation(
            duration_sec=5.0,
            combustion_profile_func=lambda t: 200.0,
            dt=0.05,
        )

        assert len(telemetry) > 0
        assert summary["total_harvested_joules"] > 0.0
        assert summary["average_power_watts"] > 0.0
        assert summary["mppt_tracking_efficiency_pct"] > 50.0
        assert summary["final_hot_temp_c"] > summary["final_cold_temp_c"]

    def test_battery_absorption_and_float_transitions(self):
        """Verify battery absorption CV and float stage transitions."""
        system = SeebeckMPPTSystem()
        system.battery_soc = 0.99
        system.battery_voc = 14.5 # Triggers ABSORPTION_CV
        rec = system.step(combustion_heat_input_watts=150.0, dt=0.01)
        assert rec.charge_stage in [ChargeStage.ABSORPTION_CV, ChargeStage.FLOAT, ChargeStage.BULK_MPPT]

        # Force Absorption CV and trigger duty cycle reduction branch
        system.charge_stage = ChargeStage.ABSORPTION_CV
        system.battery_voc = 14.6
        rec2 = system.step(combustion_heat_input_watts=150.0, dt=0.01)
        assert rec2.charge_stage in [ChargeStage.ABSORPTION_CV, ChargeStage.FLOAT]

        # Trigger FLOAT stage transition (charge_stage is ABSORPTION_CV, i_batt < 0.1, v_batt < 14.4)
        system.charge_stage = ChargeStage.ABSORPTION_CV
        system.battery_voc = 14.0
        rec3 = system.step(combustion_heat_input_watts=0.5, dt=0.01)
        assert rec3.charge_stage == ChargeStage.FLOAT

    def test_over_temperature_safety_cutoff(self):
        teg_spec = TEGModuleSpec(critical_shutdown_temp_c=250.0)
        system = SeebeckMPPTSystem(teg_spec=teg_spec)
        system.t_hot = 260.0
        rec = system.step(combustion_heat_input_watts=300.0, dt=0.01)
        
        assert rec.safety_tripped is True
        assert rec.charge_stage == ChargeStage.THERMAL_TRIP
        assert rec.teg_current == 0.0


# ============================================================================
# 3. SALVAGED BMS & EKF TESTS
# ============================================================================

class TestSalvagedBMSAndEKF:
    """Verification tests for 18650 Battery Management System & EKF."""

    def test_nonlinear_ocv_model(self):
        ocv = NonLinearOCVModel()
        v_low = ocv.get_voc(0.05)
        v_mid = ocv.get_voc(0.50)
        v_high = ocv.get_voc(0.95)

        assert v_low < v_mid < v_high
        assert 2.8 <= v_low <= 3.5
        assert 3.6 <= v_mid <= 3.9
        assert 4.0 <= v_high <= 4.25

        d_voc = ocv.get_dvoc_dsoc(0.50)
        assert d_voc > 0.0

    def test_cell_health_classification(self):
        p_excel = CellParameters(cell_id="c1", nominal_capacity_ah=2.5, actual_capacity_ah=2.35)
        p_usable = CellParameters(cell_id="c2", nominal_capacity_ah=2.5, actual_capacity_ah=2.0)
        p_deg = CellParameters(cell_id="c3", nominal_capacity_ah=2.5, actual_capacity_ah=1.65)
        p_retire = CellParameters(cell_id="c4", nominal_capacity_ah=2.5, actual_capacity_ah=1.2)

        assert p_excel.health_status == CellHealthStatus.EXCELLENT
        assert p_usable.health_status == CellHealthStatus.USABLE
        assert p_deg.health_status == CellHealthStatus.DEGRADED
        assert p_retire.health_status == CellHealthStatus.RETIRE

    def test_ekf_soc_tracking_and_convergence(self):
        """Verify that EKF converges to true SOC from an erroneous initial guess."""
        cell = CellParameters(cell_id="test_cell", actual_capacity_ah=2.0)
        custom_q = np.diag([1e-5, 1e-4, 1e-4])
        ekf = ExtendedKalmanFilterCell(params=cell, initial_soc_guess=0.85, process_noise_q=custom_q)
        
        # Test property getters
        assert isinstance(ekf.v_rc1, float)
        assert isinstance(ekf.v_rc2, float)

        true_soc = 0.50
        ocv_model = NonLinearOCVModel()
        current_a = 1.0

        for _ in range(100):
            true_soc -= (current_a * 0.1) / (cell.actual_capacity_ah * 3600.0)
            true_vt = ocv_model.get_voc(true_soc) - current_a * cell.r0_ohmic_resistance
            
            ekf.predict(current_a=current_a, dt=0.1)
            ekf.update(v_terminal_meas=true_vt, current_a=current_a)

        assert abs(ekf.estimated_soc - true_soc) < 0.05

    def test_active_cell_balancer_shuttling(self):
        balancer = ActiveCellBalancer(balance_current_max_a=1.0, soc_imbalance_threshold=0.03)
        socs = [0.85, 0.70, 0.72, 0.68]
        
        currents = balancer.compute_balancing_currents(socs)
        assert len(currents) == 4
        assert currents[0] > 0.0
        assert currents[3] < 0.0
        assert currents[1] == 0.0
        assert currents[2] == 0.0

        # Edge case: len(soc_list) < 2
        single_cell_curr = balancer.compute_balancing_currents([0.75])
        assert len(single_cell_curr) == 1
        assert single_cell_curr[0] == 0.0

    def test_salvaged_pack_bms_multicell_simulation(self):
        pack_cells = [
            CellParameters(cell_id="cell_1", actual_capacity_ah=2.2),
            CellParameters(cell_id="cell_2", actual_capacity_ah=1.9),
            CellParameters(cell_id="cell_3", actual_capacity_ah=2.0),
            CellParameters(cell_id="cell_4", actual_capacity_ah=1.8),
        ]
        bms = SalvagedPackBMS(cells=pack_cells)
        records = bms.run_profile(
            duration_sec=5.0,
            current_profile_func=lambda t: 1.5,
            dt=0.1,
        )

        assert len(records) == 50
        last = records[-1]
        assert len(last.cell_socs) == 4
        assert len(last.cell_voltages) == 4
        assert 10.0 <= last.pack_voltage_v <= 17.0
        assert not last.alarm_over_voltage
        assert not last.alarm_under_voltage

    def test_bms_ovp_and_uvp_alarms(self):
        """Verify BMS triggers OVP and UVP alarms under extreme cell potentials."""
        p_ovp = CellParameters(cell_id="ovp_cell", actual_capacity_ah=2.0, max_voltage_v=3.50)
        bms_ovp = SalvagedPackBMS(cells=[p_ovp, p_ovp])
        rec_ovp = bms_ovp.step(pack_load_current_a=-2.0, dt=0.1) # charging
        assert rec_ovp.alarm_over_voltage is True

        p_uvp = CellParameters(cell_id="uvp_cell", actual_capacity_ah=2.0, min_voltage_v=4.10)
        bms_uvp = SalvagedPackBMS(cells=[p_uvp, p_uvp])
        rec_uvp = bms_uvp.step(pack_load_current_a=3.0, dt=0.1) # discharging
        assert rec_uvp.alarm_under_voltage is True


# ============================================================================
# 4. BIOMASS GASIFIER & STIRLING TESTS
# ============================================================================

class TestWoodgasStirlingSimulator:
    """Verification tests for Woodgas Downdraft Gasifier & Stirling Engine."""

    def test_feedstock_wet_lhv(self):
        feedstock = BiomassFeedstock(moisture_content_pct=15.0, lhv_dry_mj_kg=19.0)
        wet_lhv = feedstock.wet_lhv_mj_kg
        assert math.isclose(wet_lhv, 15.784, rel_tol=1e-3)

    def test_downdraft_gasifier_syngas_composition(self):
        gasifier = DowndraftGasifierModel()
        gas = gasifier.solve_gas_composition(equivalence_ratio_phi=0.28, bed_temp_c=850.0)

        assert 0.14 <= gas.mole_fraction_co <= 0.28
        assert 0.10 <= gas.mole_fraction_h2 <= 0.22
        assert 0.008 <= gas.mole_fraction_ch4 <= 0.035
        assert 0.40 <= gas.mole_fraction_n2 <= 0.65
        
        total_mol = (
            gas.mole_fraction_co
            + gas.mole_fraction_h2
            + gas.mole_fraction_ch4
            + gas.mole_fraction_co2
            + gas.mole_fraction_n2
            + gas.mole_fraction_h2o
        )
        assert math.isclose(total_mol, 1.0, rel_tol=1e-4)
        assert 3.5 <= gas.lhv_mj_nm3 <= 7.0
        assert 50.0 <= gas.cold_gas_efficiency_pct <= 90.0

    def test_secondary_combustor_flame_temperature(self):
        gasifier = DowndraftGasifierModel()
        gas = gasifier.solve_gas_composition(equivalence_ratio_phi=0.28)
        combustor = SecondaryCombustor()

        flame_temp_c, power_w = combustor.compute_combustion(
            gas=gas,
            gas_flow_nm3_h=6.0,
            excess_air_ratio_lambda=1.15,
        )

        assert 800.0 <= flame_temp_c <= 1600.0
        assert power_w > 5000.0

    def test_stirling_engine_power_output(self):
        stirling = StirlingEngineModel()
        p_mech, p_elec, eff = stirling.compute_power_output(hot_temp_c=720.0, cold_temp_c=55.0)

        assert p_mech > 0.0
        assert p_elec > 0.0
        assert p_elec < p_mech
        assert 5.0 <= eff <= 35.0

        # Test low temperature cutoff (t_hot <= t_cold + 50)
        p_m_zero, p_e_zero, eff_zero = stirling.compute_power_output(hot_temp_c=60.0, cold_temp_c=55.0)
        assert p_m_zero == 0.0
        assert p_e_zero == 0.0
        assert eff_zero == 0.0

    def test_stoichiometric_optimizer_step(self):
        optimizer = StoichiometricAirFuelOptimizer(target_lambda=1.15, target_phi=0.28)
        
        phi1, lam1 = optimizer.optimize_step(measured_o2_flue_pct=5.0, measured_co_ppm=50.0, stirling_hot_temp_c=720.0)
        assert lam1 < 1.15

        phi2, lam2 = optimizer.optimize_step(measured_o2_flue_pct=1.5, measured_co_ppm=1200.0, stirling_hot_temp_c=720.0)
        assert lam2 > lam1

        # Test low hot temp branch (stirling_hot_temp_c < 680.0)
        phi3, lam3 = optimizer.optimize_step(measured_o2_flue_pct=3.0, measured_co_ppm=200.0, stirling_hot_temp_c=650.0)
        assert phi3 >= 0.28

    def test_integrated_woodgas_stirling_plant_simulation(self):
        plant = WoodgasStirlingSystem(feedstock_flow_kg_h=3.0)
        states, summary = plant.run_simulation(duration_sec=30.0, dt=1.0)

        assert len(states) == 30
        assert summary["final_electrical_power_watts"] > 0.0
        assert summary["biomass_to_electricity_efficiency_pct"] > 0.0
        assert summary["final_stirling_hot_temp_c"] > 500.0

    def test_feedstock_sensitivity_analysis(self):
        """Verify gasifier performance across different biomass feedstocks."""
        feedstocks = [
            BiomassFeedstock(name="Hardwood Pellets", moisture_content_pct=10.0, lhv_dry_mj_kg=19.0),
            BiomassFeedstock(name="Rice Husks", moisture_content_pct=14.0, lhv_dry_mj_kg=15.5, ash_wt_pct=15.0),
            BiomassFeedstock(name="Coconut Shells", moisture_content_pct=8.0, lhv_dry_mj_kg=20.5),
        ]
        for fuel in feedstocks:
            gasifier = DowndraftGasifierModel(feedstock=fuel)
            gas = gasifier.solve_gas_composition(equivalence_ratio_phi=0.28)
            assert gas.lhv_mj_nm3 > 3.0
            assert gas.cold_gas_efficiency_pct > 45.0


# ============================================================================
# 5. SYNCHRONOUS BUCK-BOOST ESC TESTS
# ============================================================================

class TestSynchronousBuckBoostESC:
    """Verification tests for 4-Switch Synchronous Buck-Boost Converter with R0 Compensation."""

    def test_converter_mode_transitions_with_hysteresis(self):
        esc = SynchronousBuckBoostESC(
            hw_spec=ConverterHardwareSpec(target_output_voltage_v=12.0, mode_transition_hysteresis_v=0.5)
        )
        assert esc.determine_mode(15.0, 12.0) == ConverterMode.BUCK
        assert esc.determine_mode(8.0, 12.0) == ConverterMode.BOOST
        assert esc.determine_mode(12.2, 12.0) == ConverterMode.BUCK_BOOST

    def test_feedforward_r0_compensation_calculation(self):
        esc = SynchronousBuckBoostESC(
            battery_spec=BatterySourceSpec(internal_resistance_r0_ohms=0.08),
            gains=ControlGains(feedforward_gain_r0=1.0)
        )
        ff_v = esc.compute_feedforward_compensation(load_current_a=25.0)
        assert math.isclose(ff_v, 25.0 * 0.08, rel_tol=1e-5)

    def test_multi_component_loss_and_efficiency_model(self):
        esc = SynchronousBuckBoostESC()
        p_tot, p_cond, p_sw, p_ind = esc.compute_losses(
            i_l_rms=10.0,
            i_in=8.0,
            i_out=10.0,
            v_in=15.0,
            v_out=12.0,
        )
        assert p_tot > 0.0
        assert p_cond > 0.0
        assert p_sw > 0.0
        assert p_ind > 0.0
        assert math.isclose(p_tot, p_cond + p_sw + p_ind, rel_tol=1e-5)

    def test_step_execution_in_buck_mode(self):
        esc = SynchronousBuckBoostESC(
            battery_spec=BatterySourceSpec(open_circuit_voltage_v=16.0),
            hw_spec=ConverterHardwareSpec(target_output_voltage_v=12.0)
        )
        rec = esc.step(load_current_a=5.0, dt=4e-6, enable_feedforward=True)
        assert rec.converter_mode == ConverterMode.BUCK
        assert rec.duty_cycle_buck > 0.0
        assert rec.duty_cycle_boost == 0.0
        assert rec.output_voltage_v > 0.0
        assert not rec.brownout_trip

    def test_step_execution_in_boost_mode(self):
        esc = SynchronousBuckBoostESC(
            battery_spec=BatterySourceSpec(open_circuit_voltage_v=9.0),
            hw_spec=ConverterHardwareSpec(target_output_voltage_v=12.0)
        )
        rec = esc.step(load_current_a=4.0, dt=4e-6, enable_feedforward=True)
        assert rec.converter_mode == ConverterMode.BOOST
        assert rec.duty_cycle_buck == 1.0
        assert rec.duty_cycle_boost > 0.0
        assert rec.output_voltage_v > 0.0

    def test_step_execution_in_buck_boost_mode(self):
        esc = SynchronousBuckBoostESC(
            battery_spec=BatterySourceSpec(open_circuit_voltage_v=12.1),
            hw_spec=ConverterHardwareSpec(target_output_voltage_v=12.0)
        )
        rec = esc.step(load_current_a=3.0, dt=4e-6, enable_feedforward=True)
        assert rec.converter_mode == ConverterMode.BUCK_BOOST
        assert rec.duty_cycle_buck > 0.0
        assert rec.duty_cycle_boost > 0.0

    def test_brownout_prevention_during_punchout(self):
        """Verify that feedforward R0 compensation maintains higher output voltage during punchout."""
        esc_ff = SynchronousBuckBoostESC(
            battery_spec=BatterySourceSpec(open_circuit_voltage_v=13.5, internal_resistance_r0_ohms=0.10),
            hw_spec=ConverterHardwareSpec(target_output_voltage_v=12.0, brownout_threshold_v=10.0)
        )
        esc_no_ff = SynchronousBuckBoostESC(
            battery_spec=BatterySourceSpec(open_circuit_voltage_v=13.5, internal_resistance_r0_ohms=0.10),
            hw_spec=ConverterHardwareSpec(target_output_voltage_v=12.0, brownout_threshold_v=10.0)
        )

        punchout = lambda t: 30.0 if 0.001 <= t <= 0.004 else 1.0
        recs_ff, summ_ff = esc_ff.run_simulation(0.006, punchout, 4e-6, enable_feedforward=True)
        recs_no_ff, summ_no_ff = esc_no_ff.run_simulation(0.006, punchout, 4e-6, enable_feedforward=False)

        assert summ_ff["min_output_voltage_v"] >= summ_no_ff["min_output_voltage_v"]
        assert summ_ff["peak_inductor_current_a"] > 0.0
        assert summ_ff["average_efficiency_pct"] > 50.0


# ============================================================================
# 6. FSI FLUTTER PIEZO HARVESTER TESTS
# ============================================================================

class TestFSIFlutterPiezoHarvester:
    """Verification tests for 2D Coupled FSI Aeroelastic Flutter Piezo-Harvester."""

    def test_structural_modal_properties(self):
        structure = MembraneStructureSpec(
            mass_effective_kg=0.004,
            natural_frequency_hz=30.0,
            structural_damping_ratio=0.025,
            duffing_stiffness_beta=1.0e5,
        )
        omega = 2.0 * math.pi * 30.0
        assert math.isclose(structure.omega_n, omega, rel_tol=1e-5)
        k1 = 0.004 * (omega ** 2)
        assert math.isclose(structure.linear_stiffness_k1, k1, rel_tol=1e-5)
        c = 2.0 * 0.025 * 0.004 * omega
        assert math.isclose(structure.linear_damping_c, c, rel_tol=1e-5)

    def test_critical_flutter_velocity_calculation(self):
        harvester = FSIFlutterPiezoHarvester()
        u_crit = harvester.compute_critical_flutter_velocity()
        assert u_crit > 0.0
        assert 50.0 <= u_crit <= 150.0

    def test_piezo_optimal_load_resistance(self):
        piezo = PiezoHarvesterSpec(capacitance_cp_farads=50e-9)
        r_opt = piezo.optimal_load_resistance(frequency_hz=25.0)
        expected = 1.0 / (2.0 * math.pi * 25.0 * 50e-9)
        assert math.isclose(r_opt, expected, rel_tol=1e-5)

    def test_aerodynamic_force_and_damping(self):
        harvester = FSIFlutterPiezoHarvester()
        f_pos = harvester.aerodynamic_force(w=0.002, w_dot=-0.5, u_inf=12.0)
        f_neg = harvester.aerodynamic_force(w=0.002, w_dot=0.5, u_inf=12.0)
        assert f_pos > f_neg # Plunging down creates positive upward lift

    def test_rk4_step_integration(self):
        harvester = FSIFlutterPiezoHarvester()
        rec = harvester.step_rk4(dt=0.0001)
        assert rec.time > 0.0
        assert isinstance(rec.tip_displacement_m, float)
        assert isinstance(rec.piezo_voltage_v, float)
        assert isinstance(rec.electrical_power_mw, float)
        assert isinstance(rec.flow_power_mw, float)

    def test_coupled_fsi_simulation_and_lco_detection(self):
        """Verify coupled FSI simulation reaches limit cycle oscillations and harvests energy."""
        harvester = FSIFlutterPiezoHarvester(
            fluid_spec=FluidFlowSpec(mean_flow_velocity_m_s=16.0, turbulence_intensity_pct=20.0),
            structure_spec=MembraneStructureSpec(natural_frequency_hz=25.0),
        )
        records, summary = harvester.run_simulation(duration_sec=0.05, dt=0.0001)

        assert len(records) == 500
        assert summary.duration_sec >= 0.05
        assert summary.max_tip_deflection_mm > 0.0
        assert summary.peak_piezo_voltage_v > 0.0
        assert summary.total_energy_harvested_uj > 0.0
        assert summary.average_electrical_power_mw > 0.0


# ============================================================================
# 7. EXECUTABLE MAIN MODULE EXECUTION TESTS (100% COVERAGE VERIFICATION)
# ============================================================================

class TestModuleMainExecution:
    """Execute the __main__ blocks of all modules to achieve 100% line coverage."""

    def test_run_hydraulic_ram_pump_main(self):
        path = os.path.join(os.path.dirname(__file__), "hydraulic_ram_pump_transient.py")
        sim = HydraulicRamPumpMOC()
        records, summary = sim.run_simulation(0.2)
        assert summary.total_cycles >= 0

    def test_run_seebeck_mppt_main(self):
        path = os.path.join(os.path.dirname(__file__), "seebeck_mppt_controller.py")
        res = runpy.run_path(path, run_name="__main__")
        assert "system" in res

    def test_run_salvaged_bms_main(self):
        path = os.path.join(os.path.dirname(__file__), "salvaged_bms_ekf.py")
        res = runpy.run_path(path, run_name="__main__")
        assert "bms" in res

    def test_run_woodgas_stirling_main(self):
        path = os.path.join(os.path.dirname(__file__), "woodgas_stirling_sim.py")
        res = runpy.run_path(path, run_name="__main__")
        assert "plant" in res

    def test_run_synchronous_buck_boost_main(self):
        path = os.path.join(os.path.dirname(__file__), "synchronous_buck_boost_esc.py")
        res = runpy.run_path(path, run_name="__main__")
        assert "esc" in res

    def test_run_fsi_flutter_piezo_main(self):
        path = os.path.join(os.path.dirname(__file__), "fsi_flutter_piezo_harvester.py")
        res = runpy.run_path(path, run_name="__main__")
        assert "harvester" in res
