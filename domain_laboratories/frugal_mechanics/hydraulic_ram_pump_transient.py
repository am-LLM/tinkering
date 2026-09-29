"""
Hydraulic Ram Pump Fluid Transient Solver (Method of Characteristics - MOC)
=============================================================================
Acoustic water-hammer 1D hyperbolic PDE fluid transient solver computing:
- Dynamic Joukowsky pressure shockwaves (P = rho * a * delta_v)
- Method of Characteristics (MOC) grid discretization with Courant Cr = 1.0
- Non-linear waste valve kinetic closure & reopening thresholds
- Delivery check valve actuation and air chamber spring elasticity
- Rankine and D'Aubuisson pumping efficiencies and head ratios

Zero-Electricity WASH (Water, Sanitation and Hygiene) Off-Grid Fluid Mechanics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class ValveState(Enum):
    """Operational state of the hydraulic ram pump waste/impulse valve."""
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


@dataclass
class DrivePipeSpec:
    """Drive pipe physical dimensions and acoustic properties."""
    length: float = 10.0          # Drive pipe length L (m)
    diameter: float = 0.05        # Inner diameter D (m) [e.g. 2 inch pipe]
    wave_speed: float = 1000.0    # Acoustic wave speed a (m/s) in water/steel pipe
    friction_factor: float = 0.02 # Darcy-Weisbach friction factor f
    supply_head: float = 3.5      # Drive head H_supply (m)
    fluid_density: float = 1000.0 # Fluid density rho (kg/m^3)
    gravity: float = 9.80665      # Gravitational acceleration g (m/s^2)

    @property
    def area(self) -> float:
        """Cross-sectional area of the drive pipe (m^2)."""
        return math.pi * (self.diameter ** 2) / 4.0


@dataclass
class WasteValveSpec:
    """Waste (impulse) valve kinetic and geometric parameters."""
    discharge_area: float = 0.0035       # Valve orifice effective open area (m^2)
    discharge_coeff: float = 0.75        # Orifice discharge coefficient Cd
    closure_velocity_thresh: float = 0.50# Flow velocity threshold (m/s) triggering rapid closure
    closure_time: float = 0.003          # Valve closure transit time (s) [rapid water hammer]
    reopen_differential_head: float = 0.1# Head drop below supply (m) required for reopening
    min_closed_time: float = 0.008       # Minimum duration valve remains closed during shockwave


@dataclass
class DeliveryValveSpec:
    """Delivery check valve and delivery line configuration."""
    delivery_head: float = 15.0          # Target static delivery head H_delivery (m)
    cracking_head: float = 0.2           # Check valve spring crack differential head (m)
    valve_area: float = 0.0015           # Check valve flow area (m^2)
    discharge_coeff: float = 0.70        # Check valve discharge coefficient Cd


@dataclass
class AirChamberSpec:
    """Air chamber (accumulator) elasticity and damping characteristics."""
    total_volume: float = 0.008          # Chamber total volume (m^3) [8 Liters]
    initial_air_volume: float = 0.005    # Initial air pocket volume V0 (m^3)
    atmospheric_pressure: float = 101325.0 # P_atm (Pa)
    polytropic_exponent: float = 1.2     # Polytropic gas index gamma


@dataclass
class SimulationRecord:
    """Time-series recording of hydraulic ram pump transient state."""
    time: float
    head_inlet: float
    head_valve_box: float
    velocity_valve_box: float
    flow_waste: float
    flow_delivery: float
    air_chamber_pressure_kpa: float
    waste_valve_state: ValveState
    delivery_valve_open: bool


@dataclass
class SimulationSummary:
    """Summary metrics of the hydraulic ram pump simulation run."""
    duration_sec: float
    total_cycles: int
    cycle_frequency_hz: float
    peak_transient_head_m: float
    peak_transient_pressure_kpa: float
    joukowsky_theoretical_head_m: float
    total_volume_wasted_liters: float
    total_volume_delivered_liters: float
    average_delivery_flow_lpm: float
    average_waste_flow_lpm: float
    head_ratio: float
    rankine_efficiency_pct: float
    daubuisson_efficiency_pct: float


class HydraulicRamPumpMOC:
    """
    1D Method of Characteristics (MOC) Transient Solver for Hydraulic Ram Pumps.
    """

    def __init__(
        self,
        pipe_spec: Optional[DrivePipeSpec] = None,
        waste_valve_spec: Optional[WasteValveSpec] = None,
        delivery_valve_spec: Optional[DeliveryValveSpec] = None,
        air_chamber_spec: Optional[AirChamberSpec] = None,
        num_reaches: int = 10,
    ) -> None:
        self.pipe = pipe_spec or DrivePipeSpec()
        self.waste_valve = waste_valve_spec or WasteValveSpec()
        self.delivery_valve = delivery_valve_spec or DeliveryValveSpec()
        self.air_chamber = air_chamber_spec or AirChamberSpec()

        self.num_reaches = max(4, num_reaches)
        self.num_nodes = self.num_reaches + 1

        self.dx = self.pipe.length / self.num_reaches
        self.dt = self.dx / self.pipe.wave_speed

        self.B = self.pipe.wave_speed / (self.pipe.gravity * self.pipe.area)
        self.R = (self.pipe.friction_factor * self.dx) / (
            2.0 * self.pipe.gravity * self.pipe.diameter * (self.pipe.area ** 2)
        )

        self.H = np.full(self.num_nodes, self.pipe.supply_head, dtype=np.float64)
        self.Q = np.zeros(self.num_nodes, dtype=np.float64)

        self.time = 0.0
        self.waste_valve_state = ValveState.OPEN
        self.waste_closure_timer = 0.0
        self.waste_closed_duration = 0.0
        self.delivery_valve_open = False
        
        self.air_pressure_pa = (
            self.air_chamber.atmospheric_pressure
            + self.pipe.fluid_density * self.pipe.gravity * self.delivery_valve.delivery_head
        )
        self.air_volume = self.air_chamber.initial_air_volume
        self.air_const_c = self.air_pressure_pa * (self.air_volume ** self.air_chamber.polytropic_exponent)

        self.total_wasted_volume = 0.0
        self.total_delivered_volume = 0.0
        self.peak_head_observed = self.pipe.supply_head
        self.cycle_count = 0

    def theoretical_joukowsky_head(self, delta_v: float) -> float:
        """Joukowsky water hammer head rise."""
        return (self.pipe.wave_speed * abs(delta_v)) / self.pipe.gravity

    def step(self) -> SimulationRecord:
        """Advance the MOC transient solution by one time step dt."""
        dt = self.dt
        H_new = np.zeros(self.num_nodes, dtype=np.float64)
        Q_new = np.zeros(self.num_nodes, dtype=np.float64)

        CP = self.H[:-1] + self.B * self.Q[:-1] - self.R * self.Q[:-1] * np.abs(self.Q[:-1])
        CM = self.H[1:] - self.B * self.Q[1:] + self.R * self.Q[1:] * np.abs(self.Q[1:])

        # 1. Interior Nodes
        H_new[1:-1] = 0.5 * (CP[:-1] + CM[1:])
        Q_new[1:-1] = (CP[:-1] - CM[1:]) / (2.0 * self.B)

        # 2. Upstream Boundary (x = 0): Constant Supply Reservoir
        H_new[0] = self.pipe.supply_head
        Q_new[0] = (self.pipe.supply_head - CM[0]) / self.B

        # 3. Downstream Boundary (x = L)
        CP_end = CP[-1]
        v_current = self.Q[-1] / self.pipe.area

        # State Machine Transitions
        if self.waste_valve_state == ValveState.OPEN:
            if v_current >= self.waste_valve.closure_velocity_thresh:
                self.waste_valve_state = ValveState.CLOSING
                self.waste_closure_timer = 0.0
                self.cycle_count += 1
        elif self.waste_valve_state == ValveState.CLOSING:
            self.waste_closure_timer += dt
            if self.waste_closure_timer >= self.waste_valve.closure_time:
                self.waste_valve_state = ValveState.CLOSED
                self.waste_closed_duration = 0.0
        elif self.waste_valve_state == ValveState.CLOSED:
            self.waste_closed_duration += dt
            if self.waste_closed_duration >= self.waste_valve.min_closed_time:
                if self.H[-1] <= self.pipe.supply_head or self.Q[-1] <= 0.0:
                    self.waste_valve_state = ValveState.OPEN

        if self.waste_valve_state == ValveState.OPEN:
            tau_waste = 1.0
        elif self.waste_valve_state == ValveState.CLOSING:
            tau_waste = max(0.0, 1.0 - (self.waste_closure_timer / self.waste_valve.closure_time))
        else:
            tau_waste = 0.0

        Cv_waste = tau_waste * self.waste_valve.discharge_coeff * self.waste_valve.discharge_area * math.sqrt(2.0 * self.pipe.gravity)
        Cv_del = self.delivery_valve.discharge_coeff * self.delivery_valve.valve_area * math.sqrt(2.0 * self.pipe.gravity)
        H_del_thresh = self.delivery_valve.delivery_head + self.delivery_valve.cracking_head

        # Boundary Solution:
        if CP_end > H_del_thresh and self.waste_valve_state != ValveState.OPEN:
            b_coeff = self.B * Cv_del
            c_coeff = H_del_thresh - CP_end
            disc = max(0.0, (b_coeff ** 2) - 4.0 * c_coeff)
            w = (-b_coeff + math.sqrt(disc)) / 2.0
            
            q_del_final = Cv_del * max(0.0, w)
            h_final = H_del_thresh + (w ** 2)
            
            if Cv_waste > 0.0:
                q_waste_final = Cv_waste * math.sqrt(max(0.0, h_final))
            else:
                q_waste_final = 0.0
        elif Cv_waste > 0.0:
            b_coeff = self.B * Cv_waste
            disc = (b_coeff ** 2) + 4.0 * max(0.0, CP_end)
            u = (-b_coeff + math.sqrt(disc)) / 2.0
            h_final = u ** 2
            q_waste_final = Cv_waste * u
            q_del_final = 0.0
        else:
            h_final = max(0.0, CP_end)
            q_waste_final = 0.0
            q_del_final = 0.0

        H_new[-1] = h_final
        Q_new[-1] = q_waste_final + q_del_final

        # Air Chamber Elastic Dynamics
        if q_del_final > 0.0:
            self.delivery_valve_open = True
            delta_vol = q_del_final * dt
            self.air_volume = max(0.0005, self.air_volume - delta_vol * 0.2)
            self.air_pressure_pa = self.air_const_c / (self.air_volume ** self.air_chamber.polytropic_exponent)
        else:
            self.delivery_valve_open = False
            self.air_volume += (self.air_chamber.initial_air_volume - self.air_volume) * 0.02
            self.air_pressure_pa = self.air_const_c / (self.air_volume ** self.air_chamber.polytropic_exponent)

        self.total_wasted_volume += q_waste_final * dt
        self.total_delivered_volume += q_del_final * dt
        if H_new[-1] > self.peak_head_observed:
            self.peak_head_observed = H_new[-1]

        self.H = H_new
        self.Q = Q_new
        self.time += dt

        chamber_kpa = self.air_pressure_pa / 1000.0

        return SimulationRecord(
            time=self.time,
            head_inlet=float(self.H[0]),
            head_valve_box=float(self.H[-1]),
            velocity_valve_box=float(self.Q[-1] / self.pipe.area),
            flow_waste=float(q_waste_final),
            flow_delivery=float(q_del_final),
            air_chamber_pressure_kpa=float(chamber_kpa),
            waste_valve_state=self.waste_valve_state,
            delivery_valve_open=self.delivery_valve_open,
        )

    def run_simulation(self, duration_sec: float) -> Tuple[List[SimulationRecord], SimulationSummary]:
        """Run full time-domain transient simulation for duration."""
        steps = int(math.ceil(duration_sec / self.dt))
        records: List[SimulationRecord] = []

        for _ in range(steps):
            rec = self.step()
            records.append(rec)

        head_ratio = self.delivery_valve.delivery_head / max(1e-3, self.pipe.supply_head)
        total_vol = self.total_wasted_volume + self.total_delivered_volume

        if self.total_wasted_volume > 0.0 and self.pipe.supply_head > 0.0:
            rankine_eff = (
                self.total_delivered_volume
                * (self.delivery_valve.delivery_head - self.pipe.supply_head)
            ) / (self.total_wasted_volume * self.pipe.supply_head) * 100.0
        else:
            rankine_eff = 0.0

        if total_vol > 0.0 and self.pipe.supply_head > 0.0:
            daubuisson_eff = (
                self.total_delivered_volume * self.delivery_valve.delivery_head
            ) / (total_vol * self.pipe.supply_head) * 100.0
        else:
            daubuisson_eff = 0.0

        max_vel = self.waste_valve.closure_velocity_thresh
        joukowsky_head = self.theoretical_joukowsky_head(max_vel) + self.pipe.supply_head

        summary = SimulationSummary(
            duration_sec=self.time,
            total_cycles=self.cycle_count,
            cycle_frequency_hz=self.cycle_count / max(1e-3, self.time),
            peak_transient_head_m=self.peak_head_observed,
            peak_transient_pressure_kpa=float(
                self.pipe.fluid_density * self.pipe.gravity * self.peak_head_observed / 1000.0
            ),
            joukowsky_theoretical_head_m=joukowsky_head,
            total_volume_wasted_liters=self.total_wasted_volume * 1000.0,
            total_volume_delivered_liters=self.total_delivered_volume * 1000.0,
            average_delivery_flow_lpm=(self.total_delivered_volume * 1000.0) / max(1e-3, self.time / 60.0),
            average_waste_flow_lpm=(self.total_wasted_volume * 1000.0) / max(1e-3, self.time / 60.0),
            head_ratio=head_ratio,
            rankine_efficiency_pct=float(np.clip(rankine_eff, 0.0, 100.0)),
            daubuisson_efficiency_pct=float(np.clip(daubuisson_eff, 0.0, 100.0)),
        )

        return records, summary
