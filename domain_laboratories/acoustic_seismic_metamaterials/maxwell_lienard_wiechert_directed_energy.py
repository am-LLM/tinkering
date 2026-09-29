"""
Relativistic Maxwell Liénard-Wiechert Potential & Poynting Flux Solver.

Calculates exact retarded potentials, velocity/Coulomb fields, acceleration/radiation fields,
Poynting vector flux, and relativistic angular radiated power distributions from
accelerating charges (synchrotrons, undulators, bremsstrahlung, directed energy beams).

Theoretical Foundations:
-----------------------
1. Retarded Time Root Condition:
   c * (t - t_ret) - |r_obs - r_q(t_ret)| = 0
   Unique root guaranteed by subluminal speed (beta < 1).

2. Liénard-Wiechert Potentials:
   Phi(r, t) = [ q / (4*pi*eps0 * R * (1 - n . beta)) ]_ret
   A(r, t)   = [ mu0*q*c*beta / (4*pi * R * (1 - n . beta)) ]_ret

3. Field Decomposition (Heaviside-Feynman / Liénard-Wiechert):
   E(r, t) = E_vel + E_rad
   E_vel = (q / (4*pi*eps0)) * [ (n - beta) / (gamma^2 * (1 - n.beta)^3 * R^2) ]_ret
   E_rad = (q / (4*pi*eps0*c)) * [ (n x ((n - beta) x beta_dot)) / ((1 - n.beta)^3 * R) ]_ret
   B(r, t) = (1/c) * [n]_ret x E(r, t)

4. Directed Energy Poynting Flux:
   S(r, t) = (1 / mu0) * (E x B)
   dP/dOmega(t_ret) = (q^2 / (16*pi^2*eps0*c)) * |n x ((n - beta) x beta_dot)|^2 / (1 - n.beta)^5
   P_total = (q^2 * gamma^6 / (6*pi*eps0*c)) * ( |beta_dot|^2 - |beta x beta_dot|^2 )
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import root_scalar
from typing import Callable, Dict, List, Optional, Tuple


# Fundamental Physical Constants (CODATA 2018)
C_VACUUM: float = 299_792_458.0          # Speed of light in vacuum (m/s)
EPSILON_0: float = 8.8541878128e-12     # Vacuum permittivity (F/m)
MU_0: float = 1.25663706212e-6          # Vacuum permeability (H/m)
ELEMENTARY_CHARGE: float = 1.602176634e-19 # Elementary charge e (C)
ELECTRON_MASS_KG: float = 9.1093837015e-31 # Electron rest mass (kg)


@dataclass
class RelativisticTrajectoryState:
    """Kinematic state of the radiating charge at retarded time t_ret."""
    t_ret_s: float
    pos_m: np.ndarray          # Position r_q (3,)
    vel_mps: np.ndarray        # Velocity v (3,)
    acc_mps2: np.ndarray       # Acceleration a (3,)
    beta: np.ndarray           # beta = v / c (3,)
    beta_dot: np.ndarray       # beta_dot = a / c (3,)
    gamma: float               # Lorentz factor 1 / sqrt(1 - beta^2)


@dataclass
class RadiationFieldResult:
    """Electromagnetic fields and Poynting flux at observation point (r_obs, t)."""
    t_obs_s: float
    t_ret_s: float
    r_obs_m: np.ndarray
    phi_potential_volts: float
    a_potential_t_m: np.ndarray        # Vector potential A (3,)
    e_velocity_field_v_per_m: np.ndarray # Coulomb / velocity electric field (3,)
    e_radiation_field_v_per_m: np.ndarray # Radiation electric field (3,)
    e_total_field_v_per_m: np.ndarray     # E_total = E_vel + E_rad (3,)
    b_total_field_tesla: np.ndarray       # Magnetic induction field B (3,)
    poynting_vector_w_per_m2: np.ndarray  # S = (1/mu0) * (E x B) (3,)
    poynting_flux_magnitude: float
    radiated_power_per_solid_angle_w_sr: float
    total_instantaneous_radiated_power_w: float


class RelativisticLienardWiechertSolver:
    """
    Exact Liénard-Wiechert Retarded Potential & Directed Energy Field Solver.
    """

    def __init__(self, charge_coulombs: float = ELEMENTARY_CHARGE):
        self.q = charge_coulombs

    def compute_retarded_time(
        self,
        t_obs: float,
        r_obs: np.ndarray,
        trajectory_func: Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        t_guess: Optional[float] = None
    ) -> float:
        """
        Finds exact retarded time t_ret satisfying:
        c * (t_obs - t_ret) - |r_obs - r_q(t_ret)| = 0
        """
        r_obs = np.asarray(r_obs, dtype=np.float64)

        def objective(t_ret: float) -> float:
            r_q, _, _ = trajectory_func(t_ret)
            dist = np.linalg.norm(r_obs - r_q)
            return float(C_VACUUM * (t_obs - t_ret) - dist)

        def objective_prime(t_ret: float) -> float:
            r_q, v_q, _ = trajectory_func(t_ret)
            r_diff = r_obs - r_q
            dist = max(np.linalg.norm(r_diff), 1e-15)
            n_vec = r_diff / dist
            beta_vec = v_q / C_VACUUM
            return float(-C_VACUUM * (1.0 - np.dot(n_vec, beta_vec)))

        # Guess retarded time based on observation distance
        if t_guess is None:
            r_q0, _, _ = trajectory_func(t_obs)
            d0 = np.linalg.norm(r_obs - r_q0)
            t_guess = t_obs - (d0 / C_VACUUM)

        # High-order bracketed / Newton root-finding
        # t_ret must be <= t_obs
        t_low = t_obs - 20.0 * (np.linalg.norm(r_obs) + 10.0) / C_VACUUM - 1.0
        t_high = t_obs

        # Evaluate objective at bounds; expand bounds if necessary
        f_low = objective(t_low)
        f_high = objective(t_high)
        if f_low * f_high > 0:
            t_low = t_obs - 100.0
            t_high = t_obs

        try:
            sol = root_scalar(
                f=objective,
                fprime=objective_prime,
                x0=t_guess,
                bracket=[t_low, t_high],
                method='brentq'
            )
            return float(sol.root)
        except Exception:
            # Fallback to pure Newton
            sol = root_scalar(
                f=objective,
                fprime=objective_prime,
                x0=t_guess,
                method='newton'
            )
            return float(sol.root)

    def evaluate_fields(
        self,
        t_obs: float,
        r_obs: np.ndarray,
        trajectory_func: Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> RadiationFieldResult:
        """
        Evaluates exact Liénard-Wiechert potentials, E/B fields, and Poynting flux.
        """
        r_obs = np.asarray(r_obs, dtype=np.float64)
        t_ret = self.compute_retarded_time(t_obs, r_obs, trajectory_func)
        r_q, v_q, a_q = trajectory_func(t_ret)

        r_diff = r_obs - r_q
        R_dist = np.linalg.norm(r_diff)
        if R_dist < 1e-15:
            R_dist = 1e-15
        n_unit = r_diff / R_dist

        beta_vec = v_q / C_VACUUM
        beta_sq = np.dot(beta_vec, beta_vec)
        if beta_sq >= 1.0:
            # Clip to subluminal limit for numerical stability
            beta_vec = beta_vec / (np.sqrt(beta_sq) + 1e-12) * (1.0 - 1e-8)
            beta_sq = np.dot(beta_vec, beta_vec)

        gamma = 1.0 / np.sqrt(max(1.0 - beta_sq, 1e-12))
        beta_dot = a_q / C_VACUUM

        one_minus_n_beta = 1.0 - np.dot(n_unit, beta_vec)
        if abs(one_minus_n_beta) < 1e-12:
            one_minus_n_beta = 1e-12

        # Liénard-Wiechert Potentials
        eps_factor = 4.0 * np.pi * EPSILON_0
        phi = self.q / (eps_factor * R_dist * one_minus_n_beta)
        a_vec = (beta_vec / C_VACUUM) * phi

        # Velocity Field E_vel
        # E_vel = q / (4*pi*eps0) * (n - beta) / (gamma^2 * (1 - n.beta)^3 * R^2)
        vel_num = n_unit - beta_vec
        vel_denom = (gamma ** 2) * (one_minus_n_beta ** 3) * (R_dist ** 2)
        e_vel = (self.q / eps_factor) * (vel_num / vel_denom)

        # Radiation Field E_rad
        # E_rad = q / (4*pi*eps0*c) * (n x ((n - beta) x beta_dot)) / ((1 - n.beta)^3 * R)
        cross_inner = np.cross(n_unit - beta_vec, beta_dot)
        rad_num = np.cross(n_unit, cross_inner)
        rad_denom = (one_minus_n_beta ** 3) * R_dist
        e_rad = (self.q / (eps_factor * C_VACUUM)) * (rad_num / rad_denom)

        # Total E and B fields
        e_total = e_vel + e_rad
        b_total = (1.0 / C_VACUUM) * np.cross(n_unit, e_total)

        # Poynting Vector S = (1/mu0) * (E x B)
        s_poynting = (1.0 / MU_0) * np.cross(e_total, b_total)
        s_mag = float(np.linalg.norm(s_poynting))

        # Radiated Power per Unit Solid Angle dP/dOmega
        # dP/dOmega = (q^2 / (16*pi^2*eps0*c)) * |n x ((n - beta) x beta_dot)|^2 / (1 - n.beta)^5
        rad_cross_norm_sq = np.dot(rad_num, rad_num)
        dp_domega = (self.q ** 2 / (16.0 * (np.pi ** 2) * EPSILON_0 * C_VACUUM)) * (
            rad_cross_norm_sq / (one_minus_n_beta ** 5)
        )

        # Total Instantaneous Radiated Power (Liénard Formula)
        # P = (q^2 * gamma^6 / (6*pi*eps0*c)) * ( |beta_dot|^2 - |beta x beta_dot|^2 )
        bdot_sq = np.dot(beta_dot, beta_dot)
        bx_bdot = np.cross(beta_vec, beta_dot)
        bx_bdot_sq = np.dot(bx_bdot, bx_bdot)
        lienard_term = max(bdot_sq - bx_bdot_sq, 0.0)
        p_total = (self.q ** 2 * (gamma ** 6) / (6.0 * np.pi * EPSILON_0 * C_VACUUM)) * lienard_term

        return RadiationFieldResult(
            t_obs_s=t_obs,
            t_ret_s=t_ret,
            r_obs_m=r_obs,
            phi_potential_volts=float(phi),
            a_potential_t_m=a_vec,
            e_velocity_field_v_per_m=e_vel,
            e_radiation_field_v_per_m=e_rad,
            e_total_field_v_per_m=e_total,
            b_total_field_tesla=b_total,
            poynting_vector_w_per_m2=s_poynting,
            poynting_flux_magnitude=s_mag,
            radiated_power_per_solid_angle_w_sr=float(dp_domega),
            total_instantaneous_radiated_power_w=float(p_total)
        )

    # Standard Pre-Configured Relativistic Trajectories
    @staticmethod
    def synchrotron_trajectory(
        radius_m: float = 10.0,
        energy_gev: float = 1.0,
        rest_mass_kg: float = ELECTRON_MASS_KG
    ) -> Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Ultra-relativistic circular trajectory (Synchrotron Light Source).
        """
        e_joules = energy_gev * 1.0e9 * ELEMENTARY_CHARGE
        m0_c2 = rest_mass_kg * (C_VACUUM ** 2)
        gamma = max(e_joules / m0_c2, 1.0001)
        beta = np.sqrt(1.0 - 1.0 / (gamma ** 2))
        v = beta * C_VACUUM
        omega = v / radius_m

        def trajectory(t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            theta = omega * t
            r_q = np.array([radius_m * np.cos(theta), radius_m * np.sin(theta), 0.0])
            v_q = np.array([-v * np.sin(theta), v * np.cos(theta), 0.0])
            a_q = np.array([-(v ** 2 / radius_m) * np.cos(theta), -(v ** 2 / radius_m) * np.sin(theta), 0.0])
            return r_q, v_q, a_q

        return trajectory

    @staticmethod
    def undulator_fel_trajectory(
        undulator_period_m: float = 0.03, # lambda_u = 3 cm
        k_parameter: float = 1.5,          # Undulator deflection parameter K
        gamma: float = 1000.0              # Relativistic Lorentz factor
    ) -> Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Relativistic planar undulator / Free Electron Laser (FEL) trajectory.
        """
        k_u = 2.0 * np.pi / undulator_period_m
        # Average longitudinal beta_z
        beta_z_bar = 1.0 - (1.0 + 0.5 * (k_parameter ** 2)) / (2.0 * (gamma ** 2))
        v_z = beta_z_bar * C_VACUUM
        omega_u = k_u * v_z

        def trajectory(t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            x = (k_parameter / (gamma * k_u)) * np.sin(omega_u * t)
            y = 0.0
            z = v_z * t
            
            vx = (k_parameter * C_VACUUM / gamma) * np.cos(omega_u * t)
            vy = 0.0
            vz = v_z

            ax = -(k_parameter * C_VACUUM * omega_u / gamma) * np.sin(omega_u * t)
            ay = 0.0
            az = 0.0

            return np.array([x, y, z]), np.array([vx, vy, vz]), np.array([ax, ay, az])

        return trajectory

    @staticmethod
    def linear_bremsstrahlung_trajectory(
        v0_mps: float = 0.9 * C_VACUUM,
        deceleration_mps2: float = 1e18,
        t_stop_s: float = 1e-9
    ) -> Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Relativistic 1D deceleration trajectory (Bremsstrahlung emission).
        """
        def trajectory(t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if t < 0:
                pos = np.array([0.0, 0.0, v0_mps * t])
                vel = np.array([0.0, 0.0, v0_mps])
                acc = np.array([0.0, 0.0, 0.0])
            elif t <= t_stop_s:
                vel_z = max(v0_mps - deceleration_mps2 * t, 0.0)
                pos_z = v0_mps * t - 0.5 * deceleration_mps2 * (t ** 2)
                pos = np.array([0.0, 0.0, pos_z])
                vel = np.array([0.0, 0.0, vel_z])
                acc = np.array([0.0, 0.0, -deceleration_mps2])
            else:
                pos_z = v0_mps * t_stop_s - 0.5 * deceleration_mps2 * (t_stop_s ** 2)
                pos = np.array([0.0, 0.0, pos_z])
                vel = np.array([0.0, 0.0, 0.0])
                acc = np.array([0.0, 0.0, 0.0])
            return pos, vel, acc

        return trajectory
