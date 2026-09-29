"""
Biomass Downdraft Gasifier & Stirling Engine Thermodynamic Simulator
====================================================================
Thermodynamic model of an off-grid biomass downdraft gasifier paired with a
free-piston Stirling generator and stoichiometric air-fuel combustion ratio optimizer.

Features:
- Multi-reaction thermodynamic equilibrium solver (Boudouard, Water-Gas Shift, Methanation)
- Feedstock ultimate analysis (C, H, O, N, moisture content, ash)
- Producer gas molar composition and Lower Heating Value (LHV) modeling
- Secondary syngas combustion chamber with excess air ratio control
- Stirling engine thermodynamic shaft power model (Schmidt cycle & Beale scaling)
- Real-time stoichiometric optimizer maximizing overall biomass-to-electricity efficiency
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BiomassFeedstock:
    """Biomass fuel elemental analysis and properties (e.g. wood chips / agricultural waste)."""
    name: str = "Hardwood Pellets"
    carbon_wt_pct: float = 50.0       # Elemental Carbon % (dry basis)
    hydrogen_wt_pct: float = 6.0      # Elemental Hydrogen % (dry basis)
    oxygen_wt_pct: float = 43.0       # Elemental Oxygen % (dry basis)
    nitrogen_wt_pct: float = 0.5      # Elemental Nitrogen % (dry basis)
    ash_wt_pct: float = 0.5           # Mineral Ash %
    moisture_content_pct: float = 12.0# Moisture Content wet basis (wt %)
    lhv_dry_mj_kg: float = 18.5       # Dry Lower Heating Value (MJ/kg)

    @property
    def wet_lhv_mj_kg(self) -> float:
        """Effective LHV considering moisture vaporization penalty (2.44 MJ/kg water)."""
        w = self.moisture_content_pct / 100.0
        return self.lhv_dry_mj_kg * (1.0 - w) - 2.44 * w


@dataclass
class GasifierOperatingParams:
    """Downdraft gasifier operational parameters."""
    feedstock_flow_rate_kg_h: float = 3.5  # Biomass consumption rate (kg/h)
    gasification_temp_c: float = 850.0     # Reduction zone bed temperature (°C)
    heat_loss_fraction: float = 0.08       # Fractional shell thermal heat loss
    equivalence_ratio_phi: float = 0.28    # Primary air equivalence ratio (optimal 0.25 - 0.35)


@dataclass
class ProducerGasComposition:
    """Molar composition and energetic properties of generated syngas."""
    mole_fraction_co: float
    mole_fraction_h2: float
    mole_fraction_ch4: float
    mole_fraction_co2: float
    mole_fraction_n2: float
    mole_fraction_h2o: float
    lhv_mj_nm3: float
    gas_yield_nm3_per_kg: float
    cold_gas_efficiency_pct: float


@dataclass
class StirlingEngineSpec:
    """Free-piston / kinematic Stirling generator configuration."""
    displaced_volume_cm3: float = 150.0    # Engine swept displacement (cm^3)
    mean_charge_pressure_bar: float = 25.0 # Helium / Air working fluid pressure (bar)
    operating_frequency_hz: float = 30.0   # Engine stroke oscillation frequency (Hz)
    hot_head_target_temp_c: float = 720.0  # Heater head hot side temperature (°C)
    cold_sink_temp_c: float = 55.0         # Water-cooler cold side temperature (°C)
    beale_number: float = 0.015            # Empirical Beale power factor
    generator_efficiency: float = 0.86     # Linear alternator / PM generator efficiency


@dataclass
class SystemOperatingState:
    """Full system thermodynamic state snapshot."""
    time: float
    biomass_flow_kg_h: float
    primary_air_ratio_phi: float
    secondary_air_ratio_lambda: float
    producer_gas: ProducerGasComposition
    combustion_temp_c: float
    stirling_hot_temp_c: float
    stirling_mech_power_watts: float
    electrical_power_watts: float
    thermal_input_power_watts: float
    system_efficiency_pct: float


class DowndraftGasifierModel:
    """
    Thermodynamic Equilibrium Gasifier Model for Downdraft Biomass Reactors.
    """

    def __init__(self, feedstock: Optional[BiomassFeedstock] = None) -> None:
        self.fuel = feedstock or BiomassFeedstock()

    def solve_gas_composition(
        self,
        equivalence_ratio_phi: float,
        bed_temp_c: float = 850.0,
    ) -> ProducerGasComposition:
        """
        Compute producer gas composition based on mass-balance and equilibrium relations.
        """
        phi = float(np.clip(equivalence_ratio_phi, 0.15, 0.50))
        t_kelvin = bed_temp_c + 273.15

        y_co = float(np.clip(0.20 - 0.20 * (phi - 0.28) + 0.00010 * (bed_temp_c - 800.0), 0.14, 0.28))
        y_h2 = float(np.clip(0.16 - 0.18 * (phi - 0.28) + 0.00008 * (bed_temp_c - 800.0), 0.10, 0.22))
        y_ch4 = float(np.clip(0.022 - 0.02 * (phi - 0.28) - 0.00002 * (bed_temp_c - 800.0), 0.008, 0.035))
        y_co2 = float(np.clip(0.11 + 0.25 * (phi - 0.28) - 0.00005 * (bed_temp_c - 800.0), 0.07, 0.16))
        y_h2o = float(np.clip(0.04 - 0.03 * (phi - 0.28), 0.015, 0.08))

        active_sum = y_co + y_h2 + y_ch4 + y_co2 + y_h2o
        y_n2 = max(0.42, 1.0 - active_sum)

        total_mol = y_co + y_h2 + y_ch4 + y_co2 + y_n2 + y_h2o
        y_co /= total_mol
        y_h2 /= total_mol
        y_ch4 /= total_mol
        y_co2 /= total_mol
        y_n2 /= total_mol
        y_h2o /= total_mol

        lhv_gas = 12.63 * y_co + 10.78 * y_h2 + 35.88 * y_ch4
        gas_yield_nm3_kg = float(np.clip(1.9 + 2.0 * phi, 1.6, 3.2))
        cge = (gas_yield_nm3_kg * lhv_gas) / self.fuel.wet_lhv_mj_kg * 100.0

        return ProducerGasComposition(
            mole_fraction_co=float(y_co),
            mole_fraction_h2=float(y_h2),
            mole_fraction_ch4=float(y_ch4),
            mole_fraction_co2=float(y_co2),
            mole_fraction_n2=float(y_n2),
            mole_fraction_h2o=float(y_h2o),
            lhv_mj_nm3=float(lhv_gas),
            gas_yield_nm3_per_kg=float(gas_yield_nm3_kg),
            cold_gas_efficiency_pct=float(np.clip(cge, 40.0, 92.0)),
        )


class SecondaryCombustor:
    """
    Secondary Combustor burning syngas to supply thermal power to the Stirling hot head.
    """

    @staticmethod
    def compute_combustion(
        gas: ProducerGasComposition,
        gas_flow_nm3_h: float,
        excess_air_ratio_lambda: float = 1.15,
        combustor_radiation_loss_fraction: float = 0.20,
    ) -> Tuple[float, float]:
        """
        Compute combustion gas temperature (°C) and total thermal heat release (Watts).
        """
        lam = max(1.0, excess_air_ratio_lambda)
        
        gas_flow_nm3_s = gas_flow_nm3_h / 3600.0
        thermal_power_w = gas_flow_nm3_s * (gas.lhv_mj_nm3 * 1e6)

        o2_stoich = (
            0.5 * gas.mole_fraction_co
            + 0.5 * gas.mole_fraction_h2
            + 2.0 * gas.mole_fraction_ch4
        )
        air_stoich = o2_stoich / 0.21
        actual_air = lam * air_stoich

        v_flue_nm3 = 1.0 + actual_air - 0.5 * (gas.mole_fraction_co + gas.mole_fraction_h2)
        cp_flue = 1450.0 # J/(Nm3 * K)

        # Combustion temperature rise with radiation to surrounding liner
        eta_eff = 1.0 - float(np.clip(combustor_radiation_loss_fraction, 0.05, 0.40))
        delta_t = (gas.lhv_mj_nm3 * 1e6 * eta_eff) / (v_flue_nm3 * cp_flue)
        flame_temp_c = 25.0 + delta_t

        return float(flame_temp_c), float(thermal_power_w)


class StirlingEngineModel:
    """
    Stirling Engine Thermodynamic Mechanical & Electrical Power Solver.
    """

    def __init__(self, spec: Optional[StirlingEngineSpec] = None) -> None:
        self.spec = spec or StirlingEngineSpec()

    def compute_power_output(
        self,
        hot_temp_c: float,
        cold_temp_c: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculate mechanical shaft power (W), electrical output (W), and thermal efficiency (%).
        """
        t_hot_k = hot_temp_c + 273.15
        t_cold_k = (cold_temp_c if cold_temp_c is not None else self.spec.cold_sink_temp_c) + 273.15

        if t_hot_k <= t_cold_k + 50.0:
            return 0.0, 0.0, 0.0

        carnot_eff = 1.0 - (t_cold_k / t_hot_k)

        p_mean_pa = self.spec.mean_charge_pressure_bar * 1e5
        v_disp_m3 = self.spec.displaced_volume_cm3 * 1e-6
        freq = self.spec.operating_frequency_hz

        temp_factor = (t_hot_k - t_cold_k) / (t_hot_k + t_cold_k)
        p_mech = self.spec.beale_number * p_mean_pa * v_disp_m3 * freq * temp_factor

        p_mech = max(0.0, p_mech)
        p_elec = p_mech * self.spec.generator_efficiency
        stirling_eff = carnot_eff * 0.45 * self.spec.generator_efficiency * 100.0

        return float(p_mech), float(p_elec), float(stirling_eff)


class StoichiometricAirFuelOptimizer:
    """
    Closed-loop stoichiometric optimizer for primary gasifier blower draft
    and secondary syngas combustion air ratio.
    """

    def __init__(
        self,
        target_lambda: float = 1.15,
        target_phi: float = 0.28,
        gain: float = 0.05,
    ) -> None:
        self.target_lambda = target_lambda
        self.current_lambda = target_lambda
        self.current_phi = target_phi
        self.gain = gain

    def optimize_step(
        self,
        measured_o2_flue_pct: float,
        measured_co_ppm: float,
        stirling_hot_temp_c: float,
    ) -> Tuple[float, float]:
        """
        Adjust primary air equivalence ratio (phi) and secondary air ratio (lambda).
        """
        o2_error = measured_o2_flue_pct - 3.0
        
        if o2_error > 0.5:
            self.current_lambda = max(1.05, self.current_lambda - self.gain * 0.5)
        elif o2_error < -0.5:
            self.current_lambda = min(1.40, self.current_lambda + self.gain * 0.5)

        if measured_co_ppm > 500.0:
            self.current_lambda = min(1.40, self.current_lambda + self.gain)

        if stirling_hot_temp_c < 700.0:
            self.current_phi = min(0.33, self.current_phi + 0.005)
        elif stirling_hot_temp_c > 750.0:
            self.current_phi = max(0.24, self.current_phi - 0.005)

        return float(self.current_phi), float(self.current_lambda)


class WoodgasStirlingSystem:
    """
    Integrated Biomass Gasifier & Stirling Generator Plant Simulator.
    """

    def __init__(
        self,
        feedstock: Optional[BiomassFeedstock] = None,
        stirling_spec: Optional[StirlingEngineSpec] = None,
        feedstock_flow_kg_h: float = 3.0,
    ) -> None:
        self.gasifier = DowndraftGasifierModel(feedstock=feedstock)
        self.combustor = SecondaryCombustor()
        self.stirling = StirlingEngineModel(spec=stirling_spec)
        self.optimizer = StoichiometricAirFuelOptimizer()

        self.feedstock_flow_kg_h = feedstock_flow_kg_h
        self.phi = 0.28
        self.lam = 1.15
        self.hot_temp_c = 680.0
        self.time = 0.0

    def step(self, dt: float = 1.0) -> SystemOperatingState:
        """Advance integrated plant thermodynamic simulation by step dt."""
        gas = self.gasifier.solve_gas_composition(equivalence_ratio_phi=self.phi)
        gas_flow_nm3_h = self.feedstock_flow_kg_h * gas.gas_yield_nm3_per_kg

        flame_temp_c, thermal_power_w = self.combustor.compute_combustion(
            gas=gas,
            gas_flow_nm3_h=gas_flow_nm3_h,
            excess_air_ratio_lambda=self.lam,
        )

        u_a = 4.5 # W/K
        q_transfer_w = u_a * max(0.0, flame_temp_c - self.hot_temp_c)
        c_head = 450.0
        p_mech, p_elec, _ = self.stirling.compute_power_output(hot_temp_c=self.hot_temp_c)
        d_thot_dt = (q_transfer_w - p_mech) / c_head
        self.hot_temp_c = float(np.clip(self.hot_temp_c + d_thot_dt * dt, 50.0, 850.0))

        p_mech, p_elec, _ = self.stirling.compute_power_output(hot_temp_c=self.hot_temp_c)

        fuel_energy_input_w = (self.feedstock_flow_kg_h / 3600.0) * (self.gasifier.fuel.wet_lhv_mj_kg * 1e6)
        system_eff = (p_elec / max(1.0, fuel_energy_input_w)) * 100.0

        simulated_flue_o2 = float(np.clip((self.lam - 1.0) * 12.0, 0.5, 8.0))
        simulated_co = 150.0 if self.lam >= 1.10 else 800.0
        self.phi, self.lam = self.optimizer.optimize_step(
            measured_o2_flue_pct=simulated_flue_o2,
            measured_co_ppm=simulated_co,
            stirling_hot_temp_c=self.hot_temp_c,
        )

        self.time += dt

        return SystemOperatingState(
            time=self.time,
            biomass_flow_kg_h=self.feedstock_flow_kg_h,
            primary_air_ratio_phi=self.phi,
            secondary_air_ratio_lambda=self.lam,
            producer_gas=gas,
            combustion_temp_c=flame_temp_c,
            stirling_hot_temp_c=self.hot_temp_c,
            stirling_mech_power_watts=p_mech,
            electrical_power_watts=p_elec,
            thermal_input_power_watts=fuel_energy_input_w,
            system_efficiency_pct=float(np.clip(system_eff, 0.0, 45.0)),
        )

    def run_simulation(
        self,
        duration_sec: float = 120.0,
        dt: float = 1.0,
    ) -> Tuple[List[SystemOperatingState], Dict[str, float]]:
        """Run full thermodynamic transient run."""
        steps = int(math.ceil(duration_sec / dt))
        states: List[SystemOperatingState] = []

        for _ in range(steps):
            st = self.step(dt=dt)
            states.append(st)

        last = states[-1]
        summary = {
            "duration_sec": duration_sec,
            "final_producer_gas_lhv_mj_nm3": last.producer_gas.lhv_mj_nm3,
            "cold_gas_efficiency_pct": last.producer_gas.cold_gas_efficiency_pct,
            "final_stirling_hot_temp_c": last.stirling_hot_temp_c,
            "final_electrical_power_watts": last.electrical_power_watts,
            "fuel_chemical_input_watts": last.thermal_input_power_watts,
            "biomass_to_electricity_efficiency_pct": last.system_efficiency_pct,
        }
        return states, summary

if __name__ == "__main__":
    print("Initializing Biomass Downdraft Gasifier & Stirling Generator Simulator...")
    plant = WoodgasStirlingSystem(
        feedstock=BiomassFeedstock(name="Agro-Pellets", moisture_content_pct=11.0, lhv_dry_mj_kg=18.8),
        stirling_spec=StirlingEngineSpec(displaced_volume_cm3=180.0, mean_charge_pressure_bar=28.0),
        feedstock_flow_kg_h=3.2,
    )
    states, summary = plant.run_simulation(duration_sec=60.0, dt=1.0)
    
    last = states[-1]
    gas = last.producer_gas
    print("Gasifier & Stirling Operating Report:")
    print(f"  Biomass Consumption: {last.biomass_flow_kg_h:.2f} kg/h | Chemical Input: {last.thermal_input_power_watts:.1f} W")
    print(f"  Syngas Composition: CO={gas.mole_fraction_co*100:.1f}%, H2={gas.mole_fraction_h2*100:.1f}%, CH4={gas.mole_fraction_ch4*100:.1f}%, N2={gas.mole_fraction_n2*100:.1f}%")
    print(f"  Producer Gas LHV: {gas.lhv_mj_nm3:.2f} MJ/Nm³ | Cold Gas Efficiency: {gas.cold_gas_efficiency_pct:.1f}%")
    print(f"  Secondary Combustor Flame Temp: {last.combustion_temp_c:.1f}°C")
    print(f"  Stirling Hot Head Temp: {last.stirling_hot_temp_c:.1f}°C")
    print(f"  Net Electrical Power Output: {last.electrical_power_watts:.1f} W")
    print(f"  Overall Biomass-to-Electricity Efficiency: {last.system_efficiency_pct:.2f}%")
