"""
Lithospheric Seismogenesis & 1D Fault Rate-and-State Friction Simulator.

Implements Dieterich-Ruina rate-and-state friction mechanics for fault spring-slider
systems, resolving stick-slip earthquake cycle dynamics, aseismic nucleation,
dynamic rupture instabilities, and seismic moment release.

Governing Physics:
------------------
1. Rate-and-State Constitutive Relation:
   tau(t) = sigma * mu(V, theta)
   mu(V, theta) = mu0 + a * ln(V / V0) + b * ln(theta * V0 / Dc)
   or with regularized sinh form for V -> 0:
   mu(V, theta) = a * arcsinh( (V / (2 * V0)) * exp((mu0 + b * ln(theta * V0 / Dc)) / a) )

2. State Evolution Laws:
   - Aging Law (Dieterich): d(theta)/dt = 1 - (V * theta / Dc)
   - Slip Law (Ruina):      d(theta)/dt = - (V * theta / Dc) * ln(V * theta / Dc)

3. Spring-Slider Seismomechanics:
   Quasi-dynamic with Radiation Damping:
   tau(t) = K * (V_load * t - delta) - eta * V
   (sigma * a / V + eta) * dV/dt = K * (V_load - V) - sigma * b * d(psi)/dt
   where psi = ln(theta * V0 / Dc), eta = G / (2 * c_s).

4. Seismological Metrics:
   - Recurrence interval T_recur
   - Stress drop Delta_tau = tau_peak - tau_dynamic_residual
   - Seismic Moment M0 = G * Fault_Area * Delta_u
   - Moment Magnitude Mw = (2/3) * (log10(M0) - 9.05) [SI units]
   - Energy Partition: Radiated Energy E_R, Fracture Energy E_G, Frictional Dissipation E_F
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Optional, Tuple


class StateEvolutionLaw(str, Enum):
    AGING_LAW = "aging"   # Dieterich (1979)
    SLIP_LAW = "slip"     # Ruina (1983)


@dataclass
class FaultMaterialParameters:
    """Lithospheric and frictional fault properties."""
    sigma_n_pa: float = 50.0e6     # Effective normal stress (50 MPa ~ 2-3 km crustal depth)
    shear_modulus_pa: float = 30.0e9 # Crustal shear modulus G (30 GPa)
    shear_wave_speed_mps: float = 3000.0 # S-wave speed c_s (3 km/s)
    
    # Rate-and-State parameters
    mu0: float = 0.60              # Reference friction coefficient at V0
    v0_mps: float = 1.0e-6         # Reference slip rate (1 um/s)
    a: float = 0.008               # Direct effect coefficient (velocity strengthening if a > b)
    b: float = 0.012               # Evolution effect coefficient (velocity weakening if b > a)
    dc_m: float = 1.0e-4           # Critical slip distance Dc (100 um)
    
    @property
    def is_velocity_weakening(self) -> bool:
        """True if (b - a) > 0, required for stick-slip unstable seismogenesis."""
        return (self.b - self.a) > 0

    @property
    def radiation_damping_eta(self) -> float:
        """Radiation damping impedance eta = G / (2 * c_s) in Pa·s/m."""
        return self.shear_modulus_pa / (2.0 * self.shear_wave_speed_mps)

    @property
    def critical_stiffness_k_crit(self) -> float:
        """Critical fault stiffness k_crit = sigma_n * (b - a) / Dc."""
        return self.sigma_n_pa * max(self.b - self.a, 0.0) / self.dc_m

    @property
    def critical_nucleation_length_m(self) -> float:
        """Dieterich critical earthquake nucleation length h* = G * Dc / (pi * sigma_n * (b - a))."""
        if not self.is_velocity_weakening:
            return float('inf')
        return (self.shear_modulus_pa * self.dc_m) / (np.pi * self.sigma_n_pa * (self.b - self.a))


@dataclass
class SpringSliderConfig:
    """Geometric and tectonic loading configuration."""
    v_load_mps: float = 1.0e-9     # Tectonic plate loading rate (1 nm/s ~ 31.5 mm/year)
    stiffness_k_pa_per_m: float = 1.5e9 # System stiffness K < k_crit for unstable cycles
    fault_area_m2: float = 1.0e6   # Fault patch area A (1 km^2)
    state_law: StateEvolutionLaw = StateEvolutionLaw.AGING_LAW
    v_init_mps: float = 1.0e-9     # Initial slip velocity
    theta_init_s: Optional[float] = None # Initial state (defaults to steady state at v_init)


@dataclass
class SeismicEvent:
    """Individual seismic slip event characteristics."""
    event_id: int
    t_nucleation_s: float
    t_peak_s: float
    peak_velocity_mps: float
    coseismic_slip_m: float
    stress_drop_pa: float
    seismic_moment_nm: float
    moment_magnitude_mw: float
    radiated_energy_joules: float
    fracture_energy_joules: float


@dataclass
class SeismogenesisResult:
    """Complete output history of earthquake cycle simulation."""
    time_s: np.ndarray
    slip_m: np.ndarray
    velocity_mps: np.ndarray
    state_theta_s: np.ndarray
    shear_stress_pa: np.ndarray
    events: List[SeismicEvent]
    mean_recurrence_time_s: float
    max_moment_magnitude_mw: float


class LithosphericRateStateSimulator:
    """
    Quasi-dynamic & Dynamic 1D Fault Seismogenesis Engine.
    """

    def __init__(
        self,
        material: Optional[FaultMaterialParameters] = None,
        config: Optional[SpringSliderConfig] = None
    ):
        self.mat = material or FaultMaterialParameters()
        self.cfg = config or SpringSliderConfig()

    def friction_coefficient(self, v: float, theta: float) -> float:
        """
        Calculates standard logarithmic rate-and-state friction coefficient:
        mu(V, theta) = mu0 + a * ln(V / V0) + b * ln(theta * V0 / Dc)
        """
        v_safe = max(v, 1e-15)
        th_safe = max(theta, 1e-15)
        return float(
            self.mat.mu0
            + self.mat.a * np.log(v_safe / self.mat.v0_mps)
            + self.mat.b * np.log(th_safe * self.mat.v0_mps / self.mat.dc_m)
        )

    def state_derivative(self, v: float, theta: float) -> float:
        """
        Evaluates state variable evolution d(theta)/dt according to Aging or Slip law.
        """
        v_safe = max(v, 1e-15)
        th_safe = max(theta, 1e-15)
        if self.cfg.state_law == StateEvolutionLaw.AGING_LAW:
            return 1.0 - (v_safe * th_safe / self.mat.dc_m)
        else: # SLIP_LAW
            ratio = (v_safe * th_safe) / self.mat.dc_m
            return - ratio * np.log(max(ratio, 1e-15))

    def _ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        State vector: y = [slip (m), ln(V / V0), ln(theta * V0 / Dc)]
        Using logarithmic variables (u = ln(V/V0), psi = ln(theta * V0 / Dc))
        for extreme numerical stability across 12 orders of magnitude in slip speed.
        """
        delta = y[0]
        u = y[1]       # u = ln(V / V0) => V = V0 * exp(u)
        psi = y[2]     # psi = ln(theta * V0 / Dc) => theta = (Dc / V0) * exp(psi)

        v = self.mat.v0_mps * np.exp(u)
        theta = (self.mat.dc_m / self.mat.v0_mps) * np.exp(psi)

        # State derivative
        if self.cfg.state_law == StateEvolutionLaw.AGING_LAW:
            dtheta_dt = 1.0 - (v * theta / self.mat.dc_m)
            # d(psi)/dt = (1/theta) * dtheta/dt = (V0 / (Dc * exp(psi))) - (V / Dc)
            dpsi_dt = (self.mat.v0_mps / self.mat.dc_m) * np.exp(-psi) - (v / self.mat.dc_m)
        else:
            # Slip law: d(psi)/dt = - (V / Dc) * (u + psi)
            dpsi_dt = - (v / self.mat.dc_m) * (u + psi)

        # Force balance & dynamic slip acceleration
        # (sigma * a + eta * V) * du/dt = K * (V_load - V) - sigma * b * dpsi/dt
        numerator = self.cfg.stiffness_k_pa_per_m * (self.cfg.v_load_mps - v) - (self.mat.sigma_n_pa * self.mat.b * dpsi_dt)
        denominator = self.mat.sigma_n_pa * self.mat.a + self.mat.radiation_damping_eta * v
        du_dt = numerator / max(denominator, 1e-12)

        dslip_dt = v

        return np.array([dslip_dt, du_dt, dpsi_dt])

    def simulate(
        self,
        t_max_s: float = 3.1536e7 * 100.0, # 100 years default
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_step_s: Optional[float] = None
    ) -> SeismogenesisResult:
        """
        Integrates spring-slider ODE across multi-decade earthquake cycles.
        """
        v_init = self.cfg.v_init_mps
        theta_init = self.cfg.theta_init_s or (self.mat.dc_m / v_init)
        
        u0 = np.log(v_init / self.mat.v0_mps)
        psi0 = np.log(theta_init * self.mat.v0_mps / self.mat.dc_m)
        y0 = np.array([0.0, u0, psi0])

        # Setup stiff/adaptive ODE solver (Radau or RK45)
        sol = solve_ivp(
            fun=self._ode_rhs,
            t_span=(0.0, t_max_s),
            y0=y0,
            method='Radau',
            rtol=rtol,
            atol=atol,
            max_step=max_step_s or (t_max_s / 100.0)
        )

        time = sol.t
        slip = sol.y[0]
        velocity = self.mat.v0_mps * np.exp(sol.y[1])
        theta = (self.mat.dc_m / self.mat.v0_mps) * np.exp(sol.y[2])
        
        # Calculate shear stress tau(t)
        # tau = sigma * (mu0 + a * u + b * psi)
        shear_stress = self.mat.sigma_n_pa * (
            self.mat.mu0 + self.mat.a * sol.y[1] + self.mat.b * sol.y[2]
        )

        # Detect seismic events (peak slip velocity exceeding seismic threshold ~ 1 mm/s = 1e-3 m/s)
        seismic_threshold_mps = 1.0e-3
        is_seismic = velocity > seismic_threshold_mps
        events: List[SeismicEvent] = []

        if np.any(is_seismic):
            # Identify discrete contiguous blocks of coseismic rupture
            diff_mask = np.diff(is_seismic.astype(int))
            starts = np.where(diff_mask == 1)[0]
            ends = np.where(diff_mask == -1)[0]

            # Handle edge boundaries
            if is_seismic[0]:
                starts = np.insert(starts, 0, 0)
            if is_seismic[-1]:
                ends = np.append(ends, len(is_seismic) - 1)

            for ev_idx, (idx_s, idx_e) in enumerate(zip(starts, ends)):
                segment_v = velocity[idx_s:idx_e+1]
                segment_t = time[idx_s:idx_e+1]
                segment_slip = slip[idx_s:idx_e+1]
                segment_stress = shear_stress[idx_s:idx_e+1]

                peak_idx_local = np.argmax(segment_v)
                t_peak = segment_t[peak_idx_local]
                v_peak = segment_v[peak_idx_local]
                delta_u = max(segment_slip[-1] - segment_slip[0], 1e-9)
                stress_drop = max(np.max(segment_stress) - np.min(segment_stress), 1.0)
                
                # Seismic Moment M0 = G * A * Delta_u
                m0 = self.mat.shear_modulus_pa * self.cfg.fault_area_m2 * delta_u
                # Moment Magnitude Mw = (2/3) * (log10(M0) - 9.05) [Hanks & Kanamori 1979]
                mw = (2.0 / 3.0) * (np.log10(max(m0, 1.0)) - 9.05)
                
                # Energy Partitioning
                # Radiated Energy: E_R ~ (1 / (2*G)) * Delta_tau * M0 (Orowan estimate)
                e_radiated = 0.5 * (stress_drop / self.mat.shear_modulus_pa) * m0
                # Fracture energy E_G = 0.5 * Delta_tau_eff * Delta_u * A
                e_fracture = 0.5 * stress_drop * delta_u * self.cfg.fault_area_m2

                event = SeismicEvent(
                    event_id=ev_idx + 1,
                    t_nucleation_s=segment_t[0],
                    t_peak_s=t_peak,
                    peak_velocity_mps=v_peak,
                    coseismic_slip_m=delta_u,
                    stress_drop_pa=stress_drop,
                    seismic_moment_nm=m0,
                    moment_magnitude_mw=mw,
                    radiated_energy_joules=e_radiated,
                    fracture_energy_joules=e_fracture
                )
                events.append(event)

        # Recurrence intervals
        if len(events) >= 2:
            t_peaks = [ev.t_peak_s for ev in events]
            recurrences = np.diff(t_peaks)
            mean_recurrence = float(np.mean(recurrences))
        else:
            mean_recurrence = float(t_max_s)

        max_mw = max([ev.moment_magnitude_mw for ev in events]) if events else -99.0

        return SeismogenesisResult(
            time_s=time,
            slip_m=slip,
            velocity_mps=velocity,
            state_theta_s=theta,
            shear_stress_pa=shear_stress,
            events=events,
            mean_recurrence_time_s=mean_recurrence,
            max_moment_magnitude_mw=max_mw
        )
