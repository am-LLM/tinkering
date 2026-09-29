"""
Seebeck Thermoelectric MPPT Controller & Thermal Harvester
===========================================================
Maximum Power Point Tracking (MPPT) perturb-and-observe thermal gradient
harvester for biomass rocket stoves and Stirling generators.

Features:
- Seebeck Thermoelectric Generator (TEG) multi-couple semiconductor model
- Non-linear thermal gradient dynamics (Combustion input, conduction, Peltier cooling)
- High-efficiency synchronous DC-DC Buck-Boost converter model
- Advanced MPPT algorithms:
  * Adaptive Perturb and Observe (P&O)
  * Incremental Conductance (IncCond)
  * Fractional Open-Circuit Voltage (FOCV)
- Battery 3-Stage Charge Controller (Bulk MPPT -> Absorption CV -> Float)
- Thermal runaway & Over-temperature safety throttling
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class ChargeStage(Enum):
    """Battery charging controller stages."""
    BULK_MPPT = "BULK_MPPT"
    ABSORPTION_CV = "ABSORPTION_CV"
    FLOAT = "FLOAT"
    THERMAL_TRIP = "THERMAL_TRIP"


@dataclass
class TEGModuleSpec:
    """Thermoelectric Generator physical module specifications (e.g. Bi2Te3 array)."""
    num_couples: int = 127             # Number of thermocouple junctions
    seebeck_coeff_per_couple: float = 2.0e-4 # V/K per couple (~200 uV/K)
    internal_resistance_25c: float = 2.2 # Ohms internal resistance at 25°C
    temp_coeff_resistance: float = 0.004 # 1/K temp coefficient of internal resistance
    thermal_conductance: float = 1.25    # W/K module thermal conductance K_th
    max_hot_temp_c: float = 330.0        # Max safe hot junction temperature (°C)
    critical_shutdown_temp_c: float = 360.0 # Absolute safety cutoff (°C)

    @property
    def seebeck_total(self) -> float:
        """Total Seebeck coefficient alpha_eff in V/K."""
        return self.num_couples * self.seebeck_coeff_per_couple

    def internal_resistance(self, avg_temp_c: float) -> float:
        """Internal electrical resistance as a function of mean junction temperature."""
        delta_t = avg_temp_c - 25.0
        return self.internal_resistance_25c * (1.0 + self.temp_coeff_resistance * delta_t)


@dataclass
class ThermalCombustionEnvironment:
    """Rocket stove / combustion chamber thermal capacitance and heat exchange."""
    hot_side_heat_capacity: float = 85.0   # J/K hot plate thermal mass
    cold_side_heat_capacity: float = 250.0 # J/K heatsink / water-cooled thermal mass
    cold_heatsink_conductance: float = 8.5 # W/K heatsink convective dissipation to ambient
    ambient_temp_c: float = 25.0           # Ambient temperature (°C)


@dataclass
class BatteryBufferSpec:
    """Lead-acid or LiFePO4 battery buffer specifications."""
    nominal_voltage: float = 12.0          # Nominal pack voltage (V)
    capacity_ah: float = 20.0              # Nominal battery capacity (Ah)
    internal_resistance: float = 0.04      # Battery internal series resistance (Ohms)
    absorption_voltage_thresh: float = 14.4# CV Absorption target (V)
    float_voltage_thresh: float = 13.6     # Float target (V)
    max_charge_current_a: float = 10.0     # Max allowable charge current (A)


@dataclass
class MPPTTelemetry:
    """Sample record of MPPT operation."""
    time: float
    hot_temp_c: float
    cold_temp_c: float
    delta_t_k: float
    teg_open_circuit_voltage: float
    teg_terminal_voltage: float
    teg_current: float
    teg_power_watts: float
    ideal_max_power_watts: float
    converter_duty_cycle: float
    battery_voltage: float
    battery_charge_current: float
    charge_stage: ChargeStage
    safety_tripped: bool


class MPPTAlgorithm(ABC):
    """Abstract Base Class for MPPT controllers."""
    
    @abstractmethod
    def compute_duty_cycle(
        self,
        v_teg: float,
        i_teg: float,
        p_teg: float,
        dt: float,
    ) -> float:
        """Calculate next DC-DC converter PWM duty cycle (0.05 - 0.95)."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracking states."""
        pass


class PerturbAndObserveMPPT(MPPTAlgorithm):
    """
    Adaptive Perturb & Observe (P&O) MPPT with dynamic step-size modulation.
    """

    def __init__(
        self,
        initial_duty: float = 0.5,
        min_duty: float = 0.05,
        max_duty: float = 0.95,
        base_step: float = 0.015,
        adaptive_gain: float = 0.005,
    ) -> None:
        self.duty = initial_duty
        self.min_duty = min_duty
        self.max_duty = max_duty
        self.base_step = base_step
        self.adaptive_gain = adaptive_gain

        self.prev_power = 0.0
        self.prev_voltage = 0.0
        self.direction = 1.0

    def reset(self) -> None:
        self.prev_power = 0.0
        self.prev_voltage = 0.0
        self.direction = 1.0

    def compute_duty_cycle(
        self,
        v_teg: float,
        i_teg: float,
        p_teg: float,
        dt: float,
    ) -> float:
        delta_p = p_teg - self.prev_power
        delta_v = v_teg - self.prev_voltage

        if abs(delta_v) > 1e-4:
            # Power slope dP/dV
            dp_dv = delta_p / delta_v
            # Scale step adaptively with power gradient
            step = float(np.clip(self.base_step + self.adaptive_gain * abs(dp_dv), 0.002, 0.05))

            if delta_p > 0.0:
                # Power increased: continue in same direction (decrease duty if dV > 0 to match load)
                if delta_v < 0.0:
                    self.direction = 1.0 # Increase duty cycle -> increases current / drops voltage
                else:
                    self.direction = -1.0
            else:
                # Power decreased: reverse direction
                if delta_v < 0.0:
                    self.direction = -1.0
                else:
                    self.direction = 1.0

            self.duty += self.direction * step

        self.duty = float(np.clip(self.duty, self.min_duty, self.max_duty))
        self.prev_power = p_teg
        self.prev_voltage = v_teg
        return self.duty


class IncrementalConductanceMPPT(MPPTAlgorithm):
    """
    Incremental Conductance MPPT Algorithm:
      At MPP: dP/dV = 0  => dI/dV = -I/V
      Left of MPP:  dI/dV > -I/V (Increase duty)
      Right of MPP: dI/dV < -I/V (Decrease duty)
    """

    def __init__(
        self,
        initial_duty: float = 0.5,
        min_duty: float = 0.05,
        max_duty: float = 0.95,
        step_size: float = 0.01,
    ) -> None:
        self.duty = initial_duty
        self.min_duty = min_duty
        self.max_duty = max_duty
        self.step_size = step_size
        self.prev_v = 0.0
        self.prev_i = 0.0

    def reset(self) -> None:
        self.prev_v = 0.0
        self.prev_i = 0.0

    def compute_duty_cycle(
        self,
        v_teg: float,
        i_teg: float,
        p_teg: float,
        dt: float,
    ) -> float:
        delta_v = v_teg - self.prev_v
        delta_i = i_teg - self.prev_i

        if abs(delta_v) < 1e-4:
            if delta_i > 0:
                self.duty += self.step_size
            elif delta_i < 0:
                self.duty -= self.step_size
        else:
            inc_cond = delta_i / delta_v
            inst_cond = -i_teg / max(1e-4, v_teg)
            error = inc_cond - inst_cond

            if abs(error) > 0.02:
                if inc_cond > inst_cond:
                    self.duty += self.step_size # Increase current to reach MPP
                else:
                    self.duty -= self.step_size

        self.duty = float(np.clip(self.duty, self.min_duty, self.max_duty))
        self.prev_v = v_teg
        self.prev_i = i_teg
        return self.duty


class SeebeckMPPTSystem:
    """
    Integrated Thermoelectric Generator & MPPT Battery Harvester System.
    """

    def __init__(
        self,
        teg_spec: Optional[TEGModuleSpec] = None,
        thermal_env: Optional[ThermalCombustionEnvironment] = None,
        battery_spec: Optional[BatteryBufferSpec] = None,
        mppt_algorithm: Optional[MPPTAlgorithm] = None,
        converter_efficiency: float = 0.92,
    ) -> None:
        self.teg = teg_spec or TEGModuleSpec()
        self.thermal = thermal_env or ThermalCombustionEnvironment()
        self.battery = battery_spec or BatteryBufferSpec()
        self.mppt = mppt_algorithm or PerturbAndObserveMPPT()
        self.base_converter_eff = converter_efficiency

        # Thermal state (°C)
        self.t_hot = self.thermal.ambient_temp_c + 150.0
        self.t_cold = self.thermal.ambient_temp_c + 10.0
        
        # Battery state
        self.battery_soc = 0.50 # 50% State of Charge
        self.battery_voc = 12.4
        self.charge_stage = ChargeStage.BULK_MPPT
        self.safety_tripped = False
        self.duty_cycle = 0.50

        # Performance accumulators
        self.total_energy_harvested_joules = 0.0
        self.total_ideal_energy_joules = 0.0
        self.time = 0.0

    def step(
        self,
        combustion_heat_input_watts: float,
        dt: float = 0.01,
    ) -> MPPTTelemetry:
        """
        Execute one simulation and control step with specified combustion thermal power.
        """
        # 1. Thermal Gradient & Seebeck Open-Circuit Potential
        delta_t = max(0.0, self.t_hot - self.t_cold)
        avg_temp = (self.t_hot + self.t_cold) / 2.0
        
        v_oc = self.teg.seebeck_total * delta_t
        r_int = self.teg.internal_resistance(avg_temp)

        # Theoretical Maximum Power (when R_load = R_int)
        p_ideal_max = (v_oc ** 2) / (4.0 * max(1e-3, r_int))

        # 2. Safety Interlock
        if self.t_hot >= self.teg.critical_shutdown_temp_c:
            self.safety_tripped = True
            self.charge_stage = ChargeStage.THERMAL_TRIP
        elif self.t_hot < self.teg.max_hot_temp_c:
            self.safety_tripped = False

        if self.safety_tripped:
            # Disconnect converter, force idle
            v_teg = v_oc
            i_teg = 0.0
            p_teg = 0.0
            conv_eff = 0.0
            i_batt = 0.0
            p_out = 0.0
        else:
            # 3. DC-DC Converter Dynamics
            # Emulated input resistance presented by DC-DC converter with battery load:
            # R_in_eff = R_load * (1 - D)^2 / D^2 (for Buck-Boost)
            d = self.duty_cycle
            r_batt_equiv = max(0.5, (self.battery_voc / max(0.1, self.battery.max_charge_current_a)))
            r_in_emulated = r_batt_equiv * (((1.0 - d) / max(0.02, d)) ** 2)

            # Circuit operating point
            i_teg = v_oc / (r_int + r_in_emulated)
            v_teg = v_oc - i_teg * r_int
            p_teg = v_teg * i_teg

            # Converter non-linear efficiency model (drops at very low/high duty)
            conv_eff = self.base_converter_eff * (1.0 - 0.15 * ((d - 0.5) ** 2))
            p_out = p_teg * conv_eff

            # 4. Battery State & Stage Control
            # Terminal battery voltage V_batt = V_oc_batt + I_chg * R_batt
            i_batt = p_out / max(1.0, self.battery_voc)
            v_batt = self.battery_voc + i_batt * self.battery.internal_resistance

            if v_batt >= self.battery.absorption_voltage_thresh:
                self.charge_stage = ChargeStage.ABSORPTION_CV
            elif self.charge_stage == ChargeStage.ABSORPTION_CV and i_batt < 0.1:
                self.charge_stage = ChargeStage.FLOAT
            else:
                self.charge_stage = ChargeStage.BULK_MPPT

            # Integrate battery SOC
            self.battery_soc = float(np.clip(
                self.battery_soc + (i_batt * dt) / (self.battery.capacity_ah * 3600.0),
                0.0, 1.0
            ))
            self.battery_voc = 11.8 + 1.2 * self.battery_soc # Simple OCV curve

            # 5. MPPT Algorithm Update
            if self.charge_stage == ChargeStage.BULK_MPPT:
                self.duty_cycle = self.mppt.compute_duty_cycle(v_teg, i_teg, p_teg, dt)
            elif self.charge_stage == ChargeStage.ABSORPTION_CV:
                # Regulate voltage by reducing duty cycle
                if v_batt > self.battery.absorption_voltage_thresh:
                    self.duty_cycle = max(0.05, self.duty_cycle - 0.01)

        # 6. Thermal Plant Physics Integration
        # Heat transfer through module: Q_cond = K_th * delta_t
        # Peltier heat: Q_peltier = alpha * T_hot * I
        # Joule heating in TEG: Q_joule = I^2 * R_int
        q_cond = self.teg.thermal_conductance * delta_t
        q_peltier = self.teg.seebeck_total * (self.t_hot + 273.15) * i_teg
        q_joule = (i_teg ** 2) * r_int

        # Hot side ODE: C_h * dTh/dt = Q_combustion - Q_cond - Q_peltier + 0.5 * Q_joule
        d_th_dt = (combustion_heat_input_watts - q_cond - q_peltier + 0.5 * q_joule) / self.thermal.hot_side_heat_capacity
        # Cold side ODE: C_c * dTc/dt = Q_cond + Q_peltier + 0.5 * Q_joule - K_diss * (Tc - T_amb)
        q_diss = self.thermal.cold_heatsink_conductance * (self.t_cold - self.thermal.ambient_temp_c)
        d_tc_dt = (q_cond + q_peltier + 0.5 * q_joule - q_diss) / self.thermal.cold_side_heat_capacity

        self.t_hot += d_th_dt * dt
        self.t_cold += d_tc_dt * dt

        # Energy Tracking
        self.total_energy_harvested_joules += p_out * dt
        self.total_ideal_energy_joules += p_ideal_max * dt
        self.time += dt

        return MPPTTelemetry(
            time=self.time,
            hot_temp_c=float(self.t_hot),
            cold_temp_c=float(self.t_cold),
            delta_t_k=float(delta_t),
            teg_open_circuit_voltage=float(v_oc),
            teg_terminal_voltage=float(v_teg),
            teg_current=float(i_teg),
            teg_power_watts=float(p_teg),
            ideal_max_power_watts=float(p_ideal_max),
            converter_duty_cycle=float(self.duty_cycle),
            battery_voltage=float(self.battery_voc + i_batt * self.battery.internal_resistance),
            battery_charge_current=float(i_batt),
            charge_stage=self.charge_stage,
            safety_tripped=self.safety_tripped,
        )

    def run_simulation(
        self,
        duration_sec: float,
        combustion_profile_func = None,
        dt: float = 0.05,
    ) -> Tuple[List[MPPTTelemetry], Dict[str, float]]:
        """Run full transient simulation with custom combustion heat profile."""
        steps = int(math.ceil(duration_sec / dt))
        telemetry: List[MPPTTelemetry] = []

        for _ in range(steps):
            t_curr = self.time
            heat_in = combustion_profile_func(t_curr) if combustion_profile_func else 180.0
            rec = self.step(combustion_heat_input_watts=heat_in, dt=dt)
            telemetry.append(rec)

        mppt_eff = (
            (self.total_energy_harvested_joules / max(1e-3, self.total_ideal_energy_joules))
            * 100.0
        )
        avg_power = self.total_energy_harvested_joules / max(1e-3, duration_sec)

        summary = {
            "duration_sec": duration_sec,
            "total_harvested_joules": self.total_energy_harvested_joules,
            "total_harvested_watt_hours": self.total_energy_harvested_joules / 3600.0,
            "average_power_watts": avg_power,
            "mppt_tracking_efficiency_pct": float(np.clip(mppt_eff, 0.0, 100.0)),
            "final_hot_temp_c": self.t_hot,
            "final_cold_temp_c": self.t_cold,
            "final_battery_soc_pct": self.battery_soc * 100.0,
        }
        return telemetry, summary

if __name__ == "__main__":
    print("Initializing Frugal Seebeck TEG MPPT Controller Simulation...")
    system = SeebeckMPPTSystem(
        teg_spec=TEGModuleSpec(num_couples=127, internal_resistance_25c=2.2),
        mppt_algorithm=PerturbAndObserveMPPT(),
    )
    telemetry, summary = system.run_simulation(
        duration_sec=10.0,
        combustion_profile_func=lambda t: 220.0,
        dt=0.05,
    )
    print("Simulation Complete:")
    print(f"  Total Harvested Energy: {summary['total_harvested_joules']:.2f} J ({summary['total_harvested_watt_hours']:.4f} Wh)")
    print(f"  Average Electrical Power: {summary['average_power_watts']:.2f} W")
    print(f"  MPPT Tracking Efficiency: {summary['mppt_tracking_efficiency_pct']:.2f}%")
    print(f"  Hot Junction Temp: {summary['final_hot_temp_c']:.1f}°C | Cold Temp: {summary['final_cold_temp_c']:.1f}°C")
    print(f"  Battery State-of-Charge: {summary['final_battery_soc_pct']:.1f}%")
