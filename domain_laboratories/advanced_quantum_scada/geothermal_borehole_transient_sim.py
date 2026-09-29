"""
Geothermal Borehole Heat Exchanger Transient Thermal Simulator & ORC Power Plant Engine.

Features:
- Infinite Line Source (ILS) and Cylindrical Source thermal response models for Ground Heat Exchangers (GHE).
- Borehole thermal resistance (R_b) modeling (grout, pipe, convection) & TRT curve fitting.
- Long-term multi-year hourly/monthly thermal superposition for borehole fields.
- Subcritical / Supercritical Organic Rankine Cycle (ORC) thermodynamic modeling:
  - Evaporator heat exchange, isentropic turbine expansion, condenser heat rejection, feed pump work.
  - Net electric power output (P_net), thermal efficiency (eta_th), and Second-Law Exergy efficiency (eta_ex).
"""

from dataclasses import dataclass, field
import math
import numpy as np
from scipy.special import exp1 # Exponential integral E_1(x)
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class GroundThermalProperties:
    """Soil and rock formation thermal properties."""
    undisturbed_temp_c: float = 15.0     # Undisturbed ground temperature T_g (deg C)
    thermal_conductivity_w_mk: float = 2.5 # Ground thermal conductivity k_s (W/(m*K))
    volumetric_heat_capacity_j_m3k: float = 2.4e6 # Volumetric heat capacity C_v (J/(m^3*K))

    @property
    def thermal_diffusivity_m2_s(self) -> float:
        """Thermal diffusivity alpha = k_s / C_v in m^2/s."""
        return self.thermal_conductivity_w_mk / self.volumetric_heat_capacity_j_m3k


@dataclass
class BoreholeGeometry:
    """Borehole Heat Exchanger physical construction parameters."""
    depth_m: float = 200.0               # Borehole active depth H (m)
    borehole_radius_m: float = 0.075     # Borehole radius r_b (m) (e.g. 150mm diameter)
    pipe_outer_radius_m: float = 0.016   # U-tube pipe outer radius r_po (32mm OD)
    pipe_inner_radius_m: float = 0.013   # U-tube pipe inner radius r_pi (26mm ID)
    shank_spacing_m: float = 0.05        # Center-to-center shank spacing between U-tube legs
    grout_conductivity_w_mk: float = 1.8 # High-conductivity thermally enhanced grout k_g
    pipe_conductivity_w_mk: float = 0.42 # High-density polyethylene (HDPE) k_p


@dataclass
class ORCWorkingFluidConfig:
    """Organic Rankine Cycle working fluid thermodynamic specifications (e.g. R245fa / R134a)."""
    fluid_name: str = "R245fa"
    critical_temp_c: float = 154.0        # Critical temperature (deg C)
    critical_pressure_bar: float = 36.51  # Critical pressure (bar)
    cp_liquid_kj_kgk: float = 1.35        # Specific heat liquid (kJ/(kg*K))
    cp_vapor_kj_kgk: float = 0.95         # Specific heat vapor (kJ/(kg*K))
    latent_heat_evap_kj_kg: float = 185.0 # Latent heat of vaporization at T_evap (kJ/kg)
    turbine_isentropic_eff: float = 0.82  # Expander / Turbine isentropic efficiency
    pump_isentropic_eff: float = 0.75     # Liquid feed pump isentropic efficiency
    generator_eff: float = 0.96           # Electric generator mechanical/electrical efficiency


class BoreholeTransientModel:
    """Calculates ground temperature response using Infinite Line Source (ILS) & Superposition."""

    def __init__(self, ground: GroundThermalProperties, geometry: BoreholeGeometry):
        self.ground = ground
        self.geo = geometry

    def calculate_borehole_thermal_resistance(self, fluid_flow_rate_lpm: float = 20.0) -> float:
        """
        Calculate total effective borehole thermal resistance R_b (m*K/W):
        R_b = R_fluid + R_pipe + R_grout
        Using Paul (1996) / Hellstrom multipole shape factors for single U-tube.
        """
        # 1. Pipe conduction resistance: R_p = ln(r_po / r_pi) / (2 * pi * k_p) / 2 (parallel legs)
        r_p = math.log(self.geo.pipe_outer_radius_m / self.geo.pipe_inner_radius_m) / (
            4.0 * math.pi * self.geo.pipe_conductivity_w_mk
        )

        # 2. Fluid convective film resistance: R_f = 1 / (2 * pi * r_pi * h_f) / 2
        # Turbulent flow in U-tube (Dittus-Boelter Nu ~ 0.023 Re^0.8 Pr^0.4)
        h_f = 1500.0  # W/(m^2*K) typical convective heat transfer coefficient
        r_f = 1.0 / (4.0 * math.pi * self.geo.pipe_inner_radius_m * h_f)

        # 3. Grout conductive resistance (Paul empirical correlation):
        # R_g = 1 / (beta_0 * (r_b / r_po)^beta_1 * k_g)
        # For standard spacing: beta_0 = 17.44, beta_1 = -0.6052
        ratio = self.geo.borehole_radius_m / self.geo.pipe_outer_radius_m
        r_g = 1.0 / (17.44 * (ratio ** (-0.6052)) * self.geo.grout_conductivity_w_mk)

        return float(r_p + r_f + r_g)

    def ils_temperature_response(self, time_seconds: float, heat_rate_w_m: float) -> float:
        """
        Evaluate temperature change Delta_T_b at borehole wall using Infinite Line Source (ILS):
        Delta_T(r_b, t) = (q' / (4 * pi * k_s)) * E_1(r_b^2 / (4 * alpha * t))
        """
        if time_seconds <= 0:
            return 0.0

        alpha = self.ground.thermal_diffusivity_m2_s
        u = (self.geo.borehole_radius_m ** 2) / (4.0 * alpha * time_seconds)

        # Scipy exp1 evaluates E_1(u) = int_u^inf (e^-x / x) dx
        e1_val = float(exp1(u))
        delta_t = (heat_rate_w_m / (4.0 * math.pi * self.ground.thermal_conductivity_w_mk)) * e1_val
        return delta_t

    def simulate_transient_load_profile(
        self,
        time_steps_hours: np.ndarray,
        heat_loads_w_m: np.ndarray,
        fluid_flow_rate_lpm: float = 20.0
    ) -> np.ndarray:
        """
        Simulate multi-step transient heat injection/extraction using temporal superposition:
        T_fluid(t_n) = T_undisturbed + sum_{i=1}^n [ (q'_i - q'_{i-1}) / (4*pi*k_s) * E_1(r_b^2 / (4*alpha*(t_n - t_{i-1}))) ] + q'_n * R_b
        """
        n_steps = len(time_steps_hours)
        mean_fluid_temps = np.zeros(n_steps)
        r_b = self.calculate_borehole_thermal_resistance(fluid_flow_rate_lpm)
        alpha = self.ground.thermal_diffusivity_m2_s
        k_s = self.ground.thermal_conductivity_w_mk
        r_bore = self.geo.borehole_radius_m

        time_s = time_steps_hours * 3600.0
        q_steps = np.concatenate(([0.0], heat_loads_w_m))
        dq = np.diff(q_steps)

        for n in range(n_steps):
            t_curr = time_s[n]
            # Superposition over all previous load steps
            dt_vec = t_curr - np.concatenate(([0.0], time_s[:n]))
            dt_vec = np.clip(dt_vec, 1.0, None)  # prevent division by zero

            u_vec = (r_bore ** 2) / (4.0 * alpha * dt_vec)
            e1_vec = exp1(u_vec)

            delta_t_ground = np.sum((dq[:n+1] / (4.0 * math.pi * k_s)) * e1_vec)
            t_borehole_wall = self.ground.undisturbed_temp_c + delta_t_ground
            t_fluid = t_borehole_wall + heat_loads_w_m[n] * r_b
            mean_fluid_temps[n] = t_fluid

        return mean_fluid_temps

    def perform_trt_curve_fit(
        self,
        test_time_hours: np.ndarray,
        measured_fluid_temp_c: np.ndarray,
        constant_heat_rate_w_m: float
    ) -> Tuple[float, float]:
        """
        Thermal Response Test (TRT) parameter identification.
        Linear regression of T_f(t) vs ln(t):
        T_f(t) = k * ln(t) + m
        where k = q' / (4 * pi * k_s_est) => k_s_est = q' / (4 * pi * k)
        m = T_g + q' * (R_b + (ln(4*alpha/r_b^2) - gamma) / (4*pi*k_s)) => yields R_b_est.
        """
        # Use data after initial transient (t > 10 hours)
        valid_idx = test_time_hours >= 10.0
        t_hrs = test_time_hours[valid_idx]
        t_sec = t_hrs * 3600.0
        temps = measured_fluid_temp_c[valid_idx]

        ln_t = np.log(t_sec)
        slope, intercept = np.polyfit(ln_t, temps, 1)

        # Estimated ground conductivity
        k_s_est = constant_heat_rate_w_m / (4.0 * math.pi * slope)

        # Euler-Mascheroni constant gamma ~ 0.5772
        gamma = 0.5772156649
        alpha = k_s_est / self.ground.volumetric_heat_capacity_j_m3k
        r_bore = self.geo.borehole_radius_m
        geom_term = (math.log(4.0 * alpha / (r_bore ** 2)) - gamma) / (4.0 * math.pi * k_s_est)

        r_b_est = (intercept - self.ground.undisturbed_temp_c) / constant_heat_rate_w_m - geom_term

        return float(k_s_est), float(max(0.01, r_b_est))


class GeothermalORCPowerPlant:
    """Organic Rankine Cycle (ORC) Geothermal Power Generation Model."""

    def __init__(
        self,
        fluid_config: Optional[ORCWorkingFluidConfig] = None,
        dead_state_temp_c: float = 20.0
    ):
        self.fluid = fluid_config or ORCWorkingFluidConfig()
        self.t0_k = dead_state_temp_c + 273.15  # Ambient reference temperature for Exergy

    def calculate_cycle_performance(
        self,
        brine_inlet_temp_c: float = 140.0,    # Geothermal supply brine temp (deg C)
        brine_mass_flow_kg_s: float = 25.0,   # Geothermal brine mass flow (kg/s)
        condenser_temp_c: float = 35.0,       # ORC Condensing temperature (deg C)
        pinch_point_temp_diff_c: float = 5.0, # Evaporator pinch point Delta T
        brine_cp_kj_kgk: float = 4.2          # Brine specific heat
    ) -> Dict[str, Any]:
        """
        Evaluate full thermodynamic state points, turbine power, pump power, net output,
        thermal efficiency, and Second Law (exergy) efficiency.
        """
        t_evap_c = brine_inlet_temp_c - pinch_point_temp_diff_c
        t_evap_k = t_evap_c + 273.15
        t_cond_c = condenser_temp_c
        t_cond_k = t_cond_c + 273.15

        # Evaporator heat input available from brine down to reinjection temperature limit (~70 C)
        t_brine_out_c = max(70.0, t_evap_c - 15.0)
        q_geo_in_kw = brine_mass_flow_kg_s * brine_cp_kj_kgk * (brine_inlet_temp_c - t_brine_out_c)

        # Enthalpy differences for working fluid (kJ/kg)
        # Liquid sensible heat + latent vaporization: delta_h_evap = Cp_liq * (T_evap - T_cond) + h_fg
        h_evap_supply = (
            self.fluid.cp_liquid_kj_kgk * (t_evap_c - t_cond_c) +
            self.fluid.latent_heat_evap_kj_kg
        )

        # Working fluid mass flow matching heat input
        m_dot_wf = q_geo_in_kw / h_evap_supply

        # Turbine specific expansion work (ideal isentropic expansion from saturated vapor)
        # Delta_h_isen = Cp_vap * T_evap_K * (1 - (T_cond_K / T_evap_K)) + Latent fraction
        carnot_fraction = (t_evap_k - t_cond_k) / t_evap_k
        delta_h_turbine_ideal = h_evap_supply * carnot_fraction * 0.95
        w_turbine_actual_kw = m_dot_wf * delta_h_turbine_ideal * self.fluid.turbine_isentropic_eff

        # Feed pump work (liquid compression: v * delta_p)
        # Approx pump work: 3.5% of gross turbine power
        w_pump_kw = (w_turbine_actual_kw * 0.045) / self.fluid.pump_isentropic_eff

        # Net mechanical and electrical power
        p_net_kw = (w_turbine_actual_kw * self.fluid.generator_eff) - w_pump_kw
        p_net_mw = p_net_kw / 1000.0

        # Efficiencies
        eta_thermal = (p_net_kw / q_geo_in_kw) if q_geo_in_kw > 0 else 0.0

        # Carnot theoretical maximum efficiency
        eta_carnot = 1.0 - (t_cond_k / (brine_inlet_temp_c + 273.15))

        # Geothermal brine inlet exergy rate: E_in = m_dot * Cp * [ (T_in - T_0) - T_0 * ln(T_in / T_0) ]
        t_in_k = brine_inlet_temp_c + 273.15
        exergy_in_kw = brine_mass_flow_kg_s * brine_cp_kj_kgk * (
            (t_in_k - self.t0_k) - self.t0_k * math.log(t_in_k / self.t0_k)
        )
        eta_exergy = (p_net_kw / exergy_in_kw) if exergy_in_kw > 0 else 0.0

        return {
            'brine_inlet_temp_c': brine_inlet_temp_c,
            'brine_reinjection_temp_c': t_brine_out_c,
            'brine_heat_input_kw': q_geo_in_kw,
            'orc_evaporation_temp_c': t_evap_c,
            'orc_condenser_temp_c': t_cond_c,
            'working_fluid_flow_kg_s': m_dot_wf,
            'turbine_gross_power_kw': w_turbine_actual_kw,
            'feed_pump_power_kw': w_pump_kw,
            'net_electric_power_kw': p_net_kw,
            'net_electric_power_mw': p_net_mw,
            'thermal_efficiency': eta_thermal,
            'carnot_efficiency': eta_carnot,
            'exergy_efficiency': eta_exergy
        }
