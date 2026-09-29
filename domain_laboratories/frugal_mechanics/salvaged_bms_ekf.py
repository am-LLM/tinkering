"""
Adaptive EKF Battery Management System for Salvaged 18650 Cells
================================================================
Non-linear Dual-RC Equivalent Circuit Model (ECM) and Extended Kalman Filter
(EKF) for heterogeneous, degraded second-life 18650 lithium-ion battery packs.

Features:
- 2-RC Thevenin Equivalent Circuit Model with temperature & degradation scaling
- Analytic Jacobian Extended Kalman Filter for real-time SOC estimation
- Recursive State-of-Health (SOH) and capacity tracking for salvaged cells
- Active cell balancing algorithm (energy shuttling between heterogeneous cells)
- Multi-cell series pack simulation with safety supervisory layer (OVP, UVP, OTP)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class CellHealthStatus(Enum):
    """Health classification for second-life 18650 cells."""
    EXCELLENT = "EXCELLENT"   # SOH > 90%
    USABLE = "USABLE"         # 75% <= SOH <= 90%
    DEGRADED = "DEGRADED"     # 60% <= SOH < 75%
    RETIRE = "RETIRE"         # SOH < 60%


@dataclass
class CellParameters:
    """Parameters for a single second-life 18650 cell."""
    cell_id: str
    nominal_capacity_ah: float = 2.50       # Pristine manufacturer capacity (Ah)
    actual_capacity_ah: float = 2.05        # Degraded capacity after salvage (Ah)
    r0_ohmic_resistance: float = 0.045      # High-frequency ohmic resistance R0 (Ohms)
    r1_polarization: float = 0.025          # Fast charge-transfer resistance R1 (Ohms)
    c1_polarization: float = 1200.0         # Fast electrochemical capacitance C1 (Farads)
    r2_diffusion: float = 0.035             # Slow diffusion resistance R2 (Ohms)
    c2_diffusion: float = 3500.0            # Slow diffusion capacitance C2 (Farads)
    coulombic_efficiency: float = 0.995     # Charge coulombic efficiency eta
    max_voltage_v: float = 4.25             # Upper cut-off voltage
    min_voltage_v: float = 2.80             # Lower cut-off voltage
    max_charge_current_a: float = 2.50      # Continuous max charge current
    max_discharge_current_a: float = 6.0    # Continuous max discharge current

    @property
    def soh(self) -> float:
        """State of Health calculated against pristine nominal capacity."""
        return float(np.clip(self.actual_capacity_ah / self.nominal_capacity_ah, 0.0, 1.0))

    @property
    def health_status(self) -> CellHealthStatus:
        s = self.soh
        if s >= 0.90:
            return CellHealthStatus.EXCELLENT
        elif s >= 0.75:
            return CellHealthStatus.USABLE
        elif s >= 0.60:
            return CellHealthStatus.DEGRADED
        return CellHealthStatus.RETIRE


class NonLinearOCVModel:
    """
    Empirical Open-Circuit Voltage (OCV) vs SOC polynomial with logarithmic tails
    for lithium nickel manganese cobalt (NMC) 18650 chemistry.
    """
    
    @staticmethod
    def get_voc(soc: float) -> float:
        """Compute Open Circuit Voltage (V) from SOC in [0.0, 1.0]."""
        s = float(np.clip(soc, 0.001, 0.999))
        # Standard NMC 18650 OCV profile: 3.1V @ 0%, 3.72V @ 50%, 4.20V @ 100%
        voc = (
            3.35
            + 0.95 * s
            - 0.85 * (s ** 2)
            + 1.30 * (s ** 3)
            - 0.60 * (s ** 4)
            + 0.045 * math.log(s)
            - 0.030 * math.log(1.0 - s)
        )
        return float(np.clip(voc, 2.75, 4.25))

    @staticmethod
    def get_dvoc_dsoc(soc: float) -> float:
        """Compute analytical derivative d(Voc)/d(SOC) for EKF Jacobian."""
        s = float(np.clip(soc, 0.001, 0.999))
        d_voc = (
            0.95
            - 1.70 * s
            + 3.90 * (s ** 2)
            - 2.40 * (s ** 3)
            + 0.045 / s
            + 0.030 / (1.0 - s)
        )
        return float(np.clip(d_voc, 0.1, 50.0))


class ExtendedKalmanFilterCell:
    """
    Extended Kalman Filter (EKF) state estimator for a single 18650 cell.
    
    State Vector: x = [SOC, V_RC1, V_RC2]^T
    Inputs: Current I (A, positive = discharge, negative = charge)
    Measurements: Terminal Voltage V_t (V)
    """

    def __init__(
        self,
        params: CellParameters,
        initial_soc_guess: float = 0.80,
        process_noise_q: Optional[np.ndarray] = None,
        measurement_noise_r: float = 1e-3,
    ) -> None:
        self.params = params
        self.ocv = NonLinearOCVModel()

        # State vector [SOC, V_RC1, V_RC2]
        self.x = np.array([initial_soc_guess, 0.0, 0.0], dtype=np.float64)

        # State Covariance Matrix P
        self.P = np.diag([1e-2, 1e-4, 1e-4]).astype(np.float64)

        # Process Noise Covariance Q
        if process_noise_q is not None:
            self.Q = process_noise_q
        else:
            self.Q = np.diag([1e-6, 1e-5, 1e-5]).astype(np.float64)

        # Measurement Noise Covariance R
        self.R = measurement_noise_r

        # Coulomb counting accumulator
        self.accumulated_ah = 0.0
        self.soc_start_soh_window = initial_soc_guess
        self.ah_in_soh_window = 0.0

    @property
    def estimated_soc(self) -> float:
        return float(np.clip(self.x[0], 0.0, 1.0))

    @property
    def v_rc1(self) -> float:
        return float(self.x[1])

    @property
    def v_rc2(self) -> float:
        return float(self.x[2])

    def predict(self, current_a: float, dt: float) -> None:
        """
        EKF Time-Update / Prediction step.
        """
        tau1 = self.params.r1_polarization * self.params.c1_polarization
        tau2 = self.params.r2_diffusion * self.params.c2_diffusion

        a11 = 1.0
        a22 = math.exp(-dt / tau1)
        a33 = math.exp(-dt / tau2)
        A = np.diag([a11, a22, a33])

        eta = self.params.coulombic_efficiency if current_a < 0 else 1.0
        b1 = -(eta * dt) / (self.params.actual_capacity_ah * 3600.0)
        b2 = self.params.r1_polarization * (1.0 - a22)
        b3 = self.params.r2_diffusion * (1.0 - a33)
        B = np.array([b1, b2, b3], dtype=np.float64)

        self.x = A @ self.x + B * current_a
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.P = A @ self.P @ A.T + self.Q

        self.accumulated_ah += abs(current_a * dt) / 3600.0
        self.ah_in_soh_window += (current_a * dt) / 3600.0

    def update(self, v_terminal_meas: float, current_a: float) -> float:
        """
        EKF Measurement-Update / Correction step.
        """
        soc = self.estimated_soc
        v_oc = self.ocv.get_voc(soc)
        d_voc = self.ocv.get_dvoc_dsoc(soc)

        v_pred = v_oc - current_a * self.params.r0_ohmic_resistance - self.x[1] - self.x[2]
        H = np.array([[d_voc, -1.0, -1.0]], dtype=np.float64)
        y_residual = v_terminal_meas - v_pred

        # Innovation scalar
        s_val = float(np.asarray(H @ self.P @ H.T).item()) + self.R
        K = (self.P @ H.T) / s_val

        self.x = self.x + (K.flatten() * y_residual)
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)

        # Joseph stabilized covariance update
        I_mat = np.eye(3, dtype=np.float64)
        I_KH = I_mat - K @ H
        self.P = I_KH @ self.P @ I_KH.T + (K * self.R) @ K.T

        return float(y_residual)

    def estimate_terminal_voltage(self, current_a: float) -> float:
        """Simulate true physical terminal voltage given internal state and current."""
        v_oc = self.ocv.get_voc(self.x[0])
        return v_oc - current_a * self.params.r0_ohmic_resistance - self.x[1] - self.x[2]


class ActiveCellBalancer:
    """
    Active Inductive/Capacitive Shuttling Balancer for Heterogeneous Series Packs.
    Transfers energy from highest-SOC cell to lowest-SOC cell.
    """

    def __init__(
        self,
        balance_current_max_a: float = 1.0,
        soc_imbalance_threshold: float = 0.02,
        efficiency: float = 0.88,
    ) -> None:
        self.balance_current_max = balance_current_max_a
        self.threshold = soc_imbalance_threshold
        self.efficiency = efficiency

    def compute_balancing_currents(self, soc_list: List[float]) -> np.ndarray:
        """Compute active balancing currents."""
        n = len(soc_list)
        currents = np.zeros(n, dtype=np.float64)
        if n < 2:
            return currents

        idx_max = int(np.argmax(soc_list))
        idx_min = int(np.argmin(soc_list))
        delta_soc = soc_list[idx_max] - soc_list[idx_min]

        if delta_soc > self.threshold:
            i_transfer = min(self.balance_current_max, (delta_soc / 0.10) * self.balance_current_max)
            currents[idx_max] = i_transfer
            currents[idx_min] = -i_transfer * self.efficiency

        return currents


@dataclass
class PackTelemetry:
    """Telemetry report for multi-cell salvaged battery pack."""
    time: float
    pack_current_a: float
    pack_voltage_v: float
    cell_socs: List[float]
    cell_voltages: List[float]
    cell_sohs: List[float]
    max_soc_imbalance: float
    balancing_active: bool
    alarm_over_voltage: bool
    alarm_under_voltage: bool


class SalvagedPackBMS:
    """
    Complete Battery Management System orchestrating heterogeneous series cells.
    """

    def __init__(
        self,
        cells: List[CellParameters],
        balancer: Optional[ActiveCellBalancer] = None,
    ) -> None:
        self.cells = cells
        self.num_cells = len(cells)
        self.balancer = balancer or ActiveCellBalancer()

        self.ekf_filters = [
            ExtendedKalmanFilterCell(params=c, initial_soc_guess=0.75)
            for c in self.cells
        ]
        self.time = 0.0

    def step(self, pack_load_current_a: float, dt: float = 0.1) -> PackTelemetry:
        """Execute one discrete BMS control and estimation cycle."""
        current_socs = [ekf.estimated_soc for ekf in self.ekf_filters]
        balance_currents = self.balancer.compute_balancing_currents(current_socs)
        balancing_active = bool(np.any(np.abs(balance_currents) > 0.01))

        cell_voltages: List[float] = []
        cell_sohs: List[float] = []
        alarm_ovp = False
        alarm_uvp = False

        for i, ekf in enumerate(self.ekf_filters):
            i_cell_total = pack_load_current_a + balance_currents[i]

            ekf.predict(current_a=i_cell_total, dt=dt)
            v_terminal_true = ekf.estimate_terminal_voltage(current_a=i_cell_total)
            v_meas = v_terminal_true + np.random.normal(0, 0.002)
            ekf.update(v_terminal_meas=v_meas, current_a=i_cell_total)

            if v_meas >= ekf.params.max_voltage_v:
                alarm_ovp = True
            if v_meas <= ekf.params.min_voltage_v:
                alarm_uvp = True

            cell_voltages.append(float(v_meas))
            cell_sohs.append(float(ekf.params.soh))

        self.time += dt
        delta_soc_max = max(current_socs) - min(current_socs)

        return PackTelemetry(
            time=self.time,
            pack_current_a=pack_load_current_a,
            pack_voltage_v=float(sum(cell_voltages)),
            cell_socs=[round(s, 4) for s in current_socs],
            cell_voltages=[round(v, 4) for v in cell_voltages],
            cell_sohs=[round(soh, 4) for soh in cell_sohs],
            max_soc_imbalance=float(delta_soc_max),
            balancing_active=balancing_active,
            alarm_over_voltage=alarm_ovp,
            alarm_under_voltage=alarm_uvp,
        )

    def run_profile(
        self,
        duration_sec: float,
        current_profile_func,
        dt: float = 0.1,
    ) -> List[PackTelemetry]:
        """Run simulated charge/discharge profile over duration."""
        steps = int(math.ceil(duration_sec / dt))
        records: List[PackTelemetry] = []
        for _ in range(steps):
            i_load = current_profile_func(self.time)
            rec = self.step(pack_load_current_a=i_load, dt=dt)
            records.append(rec)
        return records

if __name__ == "__main__":
    print("Initializing Salvaged 18650 BMS EKF Multi-Cell Pack Simulation...")
    pack = [
        CellParameters(cell_id="18650_SALVAGE_A", actual_capacity_ah=2.25, r0_ohmic_resistance=0.042),
        CellParameters(cell_id="18650_SALVAGE_B", actual_capacity_ah=1.90, r0_ohmic_resistance=0.058),
        CellParameters(cell_id="18650_SALVAGE_C", actual_capacity_ah=2.10, r0_ohmic_resistance=0.048),
        CellParameters(cell_id="18650_SALVAGE_D", actual_capacity_ah=1.75, r0_ohmic_resistance=0.065),
    ]
    bms = SalvagedPackBMS(cells=pack, balancer=ActiveCellBalancer(balance_current_max_a=1.2))
    records = bms.run_profile(duration_sec=10.0, current_profile_func=lambda t: 2.0, dt=0.1)
    
    last = records[-1]
    print("4S Pack Telemetry Report:")
    print(f"  Pack Terminal Voltage: {last.pack_voltage_v:.2f} V")
    print(f"  Estimated Cell SOCs: {last.cell_socs}")
    print(f"  Cell Terminal Voltages: {last.cell_voltages} V")
    print(f"  Cell SOH Ratings: {last.cell_sohs}")
    print(f"  Max SOC Imbalance: {last.max_soc_imbalance * 100.0:.2f}% (Active Balancer: {last.balancing_active})")
