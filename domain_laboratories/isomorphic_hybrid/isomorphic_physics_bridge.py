"""
Isomorphic Physics Bridge: Unified Cross-Disciplinary Invariant Mathematical Mapping Engine
========================================================================================
Translates state spaces, transfer functions, and stability eigenvalues bidirectionally between:
1. Mechanical Harmonic Systems (Mass, Damping, Spring, Force)
2. Electrical RLC Circuits (Inductance, Resistance, Capacitance, Voltage)
3. Hydraulic Fluid Transient Networks (Fluid Inertance, Hydraulic Resistance, Compliance, Pressure)
4. Macroeconomic Capital/Liquidity Diffusion (Capital Inertia, Market Friction, Liquidity Elasticity, Policy Stimulus)

Key Invariant Physics:
- Harmonic Oscillator: m d²x/dt² + c dx/dt + k x = F(t)
- RLC Circuit:         L d²q/dt² + R dq/dt + (1/C) q = V(t)
- Hydraulic Network:   I_h d²V/dt² + R_h dV/dt + (1/C_h) V = ΔP(t)
- Macroeconomic Model: M d²y/dt² + B dy/dt + K y = S(t)

Energy Conservation / Hamiltonian Invariants:
- Kinetic/Inductive Energy:    E_kin = 0.5 * Mass_equiv * (dx/dt)²
- Potential/Capacitive Energy: E_pot = 0.5 * Stiff_equiv * x²
- Dimensionless Damping Ratio: ζ = Damping_equiv / (2 * sqrt(Mass_equiv * Stiff_equiv))
- Undamped Natural Frequency:  ω_n = sqrt(Stiff_equiv / Mass_equiv)
- Resonant Quality Factor:     Q = 1 / (2 * ζ)
- Characteristic Impedance:    Z_0 = sqrt(Mass_equiv * Stiff_equiv)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from scipy import signal, integrate


class PhysicalDomain(str, Enum):
    MECHANICAL = "MECHANICAL"
    ELECTRICAL = "ELECTRICAL"
    HYDRAULIC = "HYDRAULIC"
    MACROECONOMIC = "MACROECONOMIC"


@dataclass(frozen=True)
class InvariantSpectralSignature:
    """Canonical invariant physical and spectral characteristics across all domains."""
    natural_frequency_rad_s: float
    damping_ratio: float
    quality_factor: float
    characteristic_impedance: float
    time_constant_s: float
    eigenvalues: Tuple[complex, complex]
    is_stable: bool
    is_underdamped: bool
    hamiltonian_energy_capacity: float  # Stored energy per unit generalized state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "natural_frequency_rad_s": self.natural_frequency_rad_s,
            "damping_ratio": self.damping_ratio,
            "quality_factor": self.quality_factor,
            "characteristic_impedance": self.characteristic_impedance,
            "time_constant_s": self.time_constant_s,
            "eigenvalues": [
                {"real": self.eigenvalues[0].real, "imag": self.eigenvalues[0].imag},
                {"real": self.eigenvalues[1].real, "imag": self.eigenvalues[1].imag},
            ],
            "is_stable": self.is_stable,
            "is_underdamped": self.is_underdamped,
            "hamiltonian_energy_capacity": self.hamiltonian_energy_capacity,
        }


@dataclass
class CanonicalStateSpace:
    """State-Space Representation: dx/dt = A x + B u, y = C x + D u."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    state_names: List[str]
    input_name: str
    output_name: str

    def compute_transfer_function(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert state space to Transfer Function: H(s) = num(s) / den(s)."""
        sys = signal.StateSpace(self.A, self.B, self.C, self.D)
        tf_sys = sys.to_tf()
        return tf_sys.num[0], tf_sys.den


class IsomorphicPhysicsBridge:
    """
    Unified Bi-directional Mathematical Isomorphism Engine.
    Guarantees exact spectral and dynamic isomorphism across disparate physical domains.
    """

    DOMAIN_VOCABULARY = {
        PhysicalDomain.MECHANICAL: {
            "inertia": "Mass (m, [kg])",
            "dissipation": "Damping Coefficient (c, [N*s/m])",
            "elasticity": "Spring Constant (k, [N/m])",
            "effort": "Force (F, [N])",
            "flow": "Velocity (v, [m/s])",
            "displacement": "Position (x, [m])",
        },
        PhysicalDomain.ELECTRICAL: {
            "inertia": "Inductance (L, [H])",
            "dissipation": "Resistance (R, [Ω])",
            "elasticity": "Elastance (1/C, [1/F])",
            "effort": "Voltage (V, [V])",
            "flow": "Current (i, [A])",
            "displacement": "Charge (q, [C])",
        },
        PhysicalDomain.HYDRAULIC: {
            "inertia": "Fluid Inertance (I_h, [Pa*s²/m³])",
            "dissipation": "Hydraulic Resistance (R_h, [Pa*s/m³])",
            "elasticity": "Hydraulic Elastance (1/C_h, [Pa/m³])",
            "effort": "Pressure Differential (ΔP, [Pa])",
            "flow": "Volumetric Flow Rate (Q, [m³/s])",
            "displacement": "Volume Displaced (V_fluid, [m³])",
        },
        PhysicalDomain.MACROECONOMIC: {
            "inertia": "Capital Adjustment Inertia (M_macro, [yr²/unit])",
            "dissipation": "Market Friction / Transaction Drag (B_macro, [yr/unit])",
            "elasticity": "Liquidity Return Elasticity (K_macro, [$/unit])",
            "effort": "Fiscal / Monetary Stimulus (S_t, [$])",
            "flow": "Capital Velocity / Flow Rate (dy/dt, [units/yr])",
            "displacement": "Output Gap / Capital Accumulation (y, [units])",
        },
    }

    def __init__(self, inertia: float, dissipation: float, elasticity: float, domain: PhysicalDomain = PhysicalDomain.MECHANICAL):
        """
        Initialize canonical dynamical core with positive definite parameters.
        Raises ValueError if parameters violate physical realism or positive definiteness.
        """
        if inertia <= 0.0 or math.isnan(inertia) or math.isinf(inertia):
            raise ValueError(f"Inertia parameter must be strictly positive and finite. Got {inertia}")
        if dissipation < 0.0 or math.isnan(dissipation) or math.isinf(dissipation):
            raise ValueError(f"Dissipation parameter must be non-negative and finite. Got {dissipation}")
        if elasticity <= 0.0 or math.isnan(elasticity) or math.isinf(elasticity):
            raise ValueError(f"Elasticity parameter must be strictly positive and finite. Got {elasticity}")

        self.inertia = float(inertia)
        self.dissipation = float(dissipation)
        self.elasticity = float(elasticity)
        self.source_domain = domain

    @property
    def spectral_signature(self) -> InvariantSpectralSignature:
        """Calculate canonical invariant spectral metrics."""
        m = self.inertia
        c = self.dissipation
        k = self.elasticity

        omega_n = math.sqrt(k / m)
        zeta = c / (2.0 * math.sqrt(k * m))
        quality_factor = 1.0 / (2.0 * zeta) if zeta > 1e-12 else float("inf")
        z_0 = math.sqrt(k * m)
        time_constant = 1.0 / (zeta * omega_n) if (zeta * omega_n) > 1e-12 else float("inf")

        discriminant = (c / m) ** 2 - 4.0 * (k / m)
        if discriminant >= 0:
            eig1 = complex(-c / (2.0 * m) + math.sqrt(discriminant) / 2.0, 0.0)
            eig2 = complex(-c / (2.0 * m) - math.sqrt(discriminant) / 2.0, 0.0)
            is_underdamped = False
        else:
            real_part = -c / (2.0 * m)
            imag_part = math.sqrt(-discriminant) / 2.0
            eig1 = complex(real_part, imag_part)
            eig2 = complex(real_part, -imag_part)
            is_underdamped = True

        is_stable = (eig1.real <= 0.0) and (eig2.real <= 0.0)
        energy_capacity = 0.5 * k

        return InvariantSpectralSignature(
            natural_frequency_rad_s=omega_n,
            damping_ratio=zeta,
            quality_factor=quality_factor,
            characteristic_impedance=z_0,
            time_constant_s=time_constant,
            eigenvalues=(eig1, eig2),
            is_stable=is_stable,
            is_underdamped=is_underdamped,
            hamiltonian_energy_capacity=energy_capacity,
        )

    def to_state_space(self) -> CanonicalStateSpace:
        """
        Generates canonical companion State Space matrices:
        State vector: x = [displacement, velocity]^T
        Input u = effort (Force, Voltage, Pressure, Stimulus)
        Output y = displacement (Position, Charge, Volume, Output Gap)
        """
        m = self.inertia
        c = self.dissipation
        k = self.elasticity

        A = np.array([
            [0.0, 1.0],
            [-k / m, -c / m]
        ], dtype=np.float64)

        B = np.array([
            [0.0],
            [1.0 / m]
        ], dtype=np.float64)

        C = np.array([[1.0, 0.0]], dtype=np.float64)
        D = np.array([[0.0]], dtype=np.float64)

        names = self.DOMAIN_VOCABULARY[self.source_domain]
        return CanonicalStateSpace(
            A=A,
            B=B,
            C=C,
            D=D,
            state_names=[names["displacement"], names["flow"]],
            input_name=names["effort"],
            output_name=names["displacement"],
        )

    def map_to_target_domain(self, target_domain: PhysicalDomain, scaling_factor: float = 1.0) -> Dict[str, Any]:
        """
        Translates parameters into target domain variables while strictly preserving
        dimensionless invariants: Damping ratio ζ, Quality factor Q, and normalized spectral response.
        """
        if target_domain not in self.DOMAIN_VOCABULARY:
            raise ValueError(f"Unsupported target domain: {target_domain}")
        if scaling_factor <= 0.0:
            raise ValueError(f"Scaling factor must be positive, got {scaling_factor}")

        spec = self.spectral_signature
        m_target = self.inertia * scaling_factor
        k_target = self.elasticity * scaling_factor
        c_target = self.dissipation * scaling_factor

        target_vocab = self.DOMAIN_VOCABULARY[target_domain]

        if target_domain == PhysicalDomain.MECHANICAL:
            domain_specific = {
                "mass_kg": m_target,
                "damping_c_Ns_m": c_target,
                "spring_k_N_m": k_target,
            }
        elif target_domain == PhysicalDomain.ELECTRICAL:
            domain_specific = {
                "inductance_L_Henry": m_target,
                "resistance_R_Ohm": c_target,
                "capacitance_C_Farad": 1.0 / k_target if k_target > 0 else float("inf"),
            }
        elif target_domain == PhysicalDomain.HYDRAULIC:
            domain_specific = {
                "fluid_inertance_I_h": m_target,
                "hydraulic_resistance_R_h": c_target,
                "hydraulic_capacitance_C_h": 1.0 / k_target if k_target > 0 else float("inf"),
            }
        else:  # PhysicalDomain.MACROECONOMIC
            domain_specific = {
                "capital_inertia_M": m_target,
                "market_friction_B": c_target,
                "liquidity_elasticity_K": k_target,
            }

        return {
            "source_domain": self.source_domain.value,
            "target_domain": target_domain.value,
            "domain_parameters": domain_specific,
            "vocabulary": target_vocab,
            "spectral_signature": spec.to_dict(),
        }

    def simulate_transient_response(
        self,
        t_span: Tuple[float, float],
        initial_state: Tuple[float, float] = (1.0, 0.0),
        forcing_function: Optional[callable] = None,
        num_points: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """
        Solves the ODE dx/dt = A x + B u(t) using high-precision Runge-Kutta 45 integration.
        """
        ss = self.to_state_space()
        A = ss.A
        B = ss.B

        t_eval = np.linspace(t_span[0], t_span[1], num_points)

        def system_dynamics(t, x):
            u = forcing_function(t) if forcing_function else 0.0
            return (A @ x + (B * u).flatten())

        sol = integrate.solve_ivp(
            system_dynamics,
            t_span,
            initial_state,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-9,
            atol=1e-12,
        )

        displacement = sol.y[0]
        velocity = sol.y[1]

        kinetic_energy = 0.5 * self.inertia * (velocity ** 2)
        potential_energy = 0.5 * self.elasticity * (displacement ** 2)
        total_energy = kinetic_energy + potential_energy

        return {
            "time": sol.t,
            "displacement": displacement,
            "velocity": velocity,
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "total_energy": total_energy,
            "success": sol.success,
        }

    def solve_diffusion_isomorphism(
        self,
        diffusion_coeff: float,
        length: float,
        nx: int = 100,
        nt: int = 500,
        t_max: float = 1.0,
    ) -> np.ndarray:
        """
        Unified 1D Diffusion Invariant Solver: ∂u/∂t = α ∂²u/∂x²
        """
        if diffusion_coeff <= 0.0:
            raise ValueError(f"Diffusion coefficient must be positive, got {diffusion_coeff}")

        dx = length / (nx - 1)
        dt = t_max / nt

        cfl = diffusion_coeff * dt / (dx ** 2)
        if cfl > 0.5:
            dt = 0.45 * (dx ** 2) / diffusion_coeff
            nt = int(math.ceil(t_max / dt))

        u = np.zeros((nt, nx), dtype=np.float64)
        x = np.linspace(0, length, nx)
        u[0, :] = np.exp(-((x - length / 2.0) ** 2) / (2.0 * (length / 10.0) ** 2))

        for n in range(0, nt - 1):
            u[n + 1, 1:-1] = u[n, 1:-1] + (diffusion_coeff * dt / (dx ** 2)) * (
                u[n, 2:] - 2.0 * u[n, 1:-1] + u[n, :-2]
            )
            u[n + 1, 0] = u[n + 1, 1]
            u[n + 1, -1] = u[n + 1, -2]

        return u