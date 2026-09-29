"""
Microalgae Photobioreactor Irradiance & Nutrient Kinetics Model
Coupled with Continuous Hydrothermal Liquefaction (HTL) Bio-crude Oil Estimation.

Implements:
1. Steele photoinhibition irradiance model with Beer-Lambert optical attenuation.
2. Dual-nutrient Monod/Droop growth kinetics (Nitrogen & Phosphorus) with lipid accumulation.
3. Dynamic chemostat/batch photobioreactor ODE integration.
4. Continuous Hydrothermal Liquefaction (HTL) bio-crude yield, HHV, and energy recovery engine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PhotobioreactorParams:
    """Kinetic and operational parameters for microalgae photobioreactor."""
    # Maximum specific growth rate (1/day)
    mu_max: float = 1.8
    # Optimal light intensity (micromol photons / m^2 / s)
    i_opt: float = 220.0
    # Biomass light extinction coefficient (m^2 / g)
    k_extinction: float = 0.12
    # Optical path length / reactor depth (m)
    depth_m: float = 0.05
    # Maintenance / respiration decay rate (1/day)
    m_decay: float = 0.05
    # Half-saturation constant for Nitrogen (g N / m^3 or mg/L)
    k_nitrogen: float = 3.5
    # Half-saturation constant for Phosphorus (g P / m^3 or mg/L)
    k_phosphorus: float = 0.4
    # Nitrogen yield requirement (g N / g biomass)
    yield_n_per_x: float = 0.075
    # Phosphorus yield requirement (g P / g biomass)
    yield_p_per_x: float = 0.009
    # Basal lipid fraction under nutrient-replete conditions (g lipid / g dry biomass)
    basal_lipid_fraction: float = 0.20
    # Maximum lipid fraction under N-stress conditions (g lipid / g dry biomass)
    max_lipid_fraction: float = 0.55
    # Basal protein fraction
    basal_protein_fraction: float = 0.45
    # Basal carbohydrate fraction
    basal_carb_fraction: float = 0.25
    # Basal ash fraction
    ash_fraction: float = 0.10


@dataclass
class HTLReactorParams:
    """Operational parameters for continuous Hydrothermal Liquefaction (HTL) unit."""
    # Operating temperature in Celsius (280 - 380 °C)
    temperature_c: float = 340.0
    # Operating pressure in MPa (10 - 25 MPa)
    pressure_mpa: float = 18.0
    # Residence time in minutes (10 - 60 min)
    residence_time_min: float = 30.0
    # Dry matter slurry concentration (wt% dry biomass, e.g. 15%)
    slurry_dry_wt_fraction: float = 0.15
    # Component biocrude conversion coefficients (empirical/mechanistic)
    conv_lipid: float = 0.92
    conv_protein: float = 0.42
    conv_carbohydrate: float = 0.22


@dataclass
class PBRSimulationState:
    """Time-series state trajectory for photobioreactor."""
    time_days: np.ndarray
    biomass_g_l: np.ndarray
    nitrogen_mg_l: np.ndarray
    phosphorus_mg_l: np.ndarray
    lipid_fraction: np.ndarray
    protein_fraction: np.ndarray
    carb_fraction: np.ndarray
    average_irradiance: np.ndarray
    growth_rate_mu: np.ndarray


@dataclass
class HTLYieldResult:
    """Output results for continuous Hydrothermal Liquefaction process."""
    biocrude_yield_wt_pct: float
    aqueous_yield_wt_pct: float
    gas_yield_wt_pct: float
    solid_char_yield_wt_pct: float
    biocrude_hhv_mj_kg: float
    feedstock_hhv_mj_kg: float
    energy_recovery_pct: float
    carbon_recovery_pct: float
    elemental_biocrude_c: float
    elemental_biocrude_h: float
    elemental_biocrude_n: float
    elemental_biocrude_o: float


class MicroalgaePhotobioreactor:
    """
    Physicochemical photobioreactor simulator coupling light attenuation,
    Steele photoinhibition, and dual Monod nutrient uptake.
    """

    def __init__(self, params: Optional[PhotobioreactorParams] = None):
        self.params = params or PhotobioreactorParams()

    def calculate_average_irradiance(self, i_surface: float, biomass_x: float) -> float:
        """
        Calculates average irradiance inside reactor depth using Beer-Lambert law.
        I_avg = I0 / (ka * X * L) * (1 - exp(-ka * X * L))
        """
        if i_surface <= 0.0:
            return 0.0
        x = max(1e-6, biomass_x)
        optical_thickness = self.params.k_extinction * x * self.params.depth_m
        if optical_thickness < 1e-5:
            return i_surface
        i_avg = (i_surface / optical_thickness) * (1.0 - np.exp(-optical_thickness))
        return float(i_avg)

    def calculate_growth_rate(
        self, i_avg: float, nitrogen_n: float, phosphorus_p: float
    ) -> float:
        """
        Computes specific growth rate mu using Steele photoinhibition and Liebig's law of the minimum for N & P.
        mu_I = mu_max * (I / I_opt) * exp(1 - I / I_opt)
        f_N = N / (K_N + N)
        f_P = P / (K_P + P)
        """
        if i_avg <= 1e-4:
            return -self.params.m_decay

        # Steele photoinhibition
        i_ratio = i_avg / self.params.i_opt
        mu_light = self.params.mu_max * i_ratio * np.exp(1.0 - i_ratio)

        # Nutrient limitations (Monod)
        f_n = max(0.0, nitrogen_n) / (self.params.k_nitrogen + max(0.0, nitrogen_n))
        f_p = max(0.0, phosphorus_p) / (self.params.k_phosphorus + max(0.0, phosphorus_p))

        nutrient_factor = min(f_n, f_p)
        mu_gross = mu_light * nutrient_factor
        mu_net = mu_gross - self.params.m_decay
        return float(mu_net)

    def calculate_biochemical_composition(
        self, nitrogen_n: float
    ) -> Tuple[float, float, float, float]:
        """
        Computes (lipid_frac, protein_frac, carb_frac, ash_frac) based on nitrogen stress.
        """
        # Nitrogen stress factor: 1.0 (replete) -> 0.0 (fully starved)
        stress_ratio = max(0.0, nitrogen_n) / (self.params.k_nitrogen * 2.0 + max(0.0, nitrogen_n))
        starvation_factor = 1.0 - stress_ratio

        lipid = self.params.basal_lipid_fraction + (
            self.params.max_lipid_fraction - self.params.basal_lipid_fraction
        ) * starvation_factor

        ash = self.params.ash_fraction
        rem = max(0.0, 1.0 - lipid - ash)
        protein = rem * (self.params.basal_protein_fraction / (self.params.basal_protein_fraction + self.params.basal_carb_fraction))
        carb = rem - protein

        return float(lipid), float(protein), float(carb), float(ash)

    def simulate(
        self,
        t_days: float = 14.0,
        dt_days: float = 0.05,
        initial_x: float = 0.1,         # g/L
        initial_n: float = 80.0,        # mg/L
        initial_p: float = 10.0,        # mg/L
        i_surface: float = 350.0,       # micromol photons/m^2/s
        dilution_rate: float = 0.0,     # 1/day (0.0 for batch, >0 for chemostat)
        inlet_n: float = 80.0,          # mg/L in feed
        inlet_p: float = 10.0,          # mg/L in feed
    ) -> PBRSimulationState:
        """
        Simulates photobioreactor trajectory via 4th-order Runge-Kutta (RK4) numerical integration.
        """
        if t_days <= 0 or dt_days <= 0:
            raise ValueError("Simulation times must be strictly positive.")

        steps = int(np.ceil(t_days / dt_days)) + 1
        time_arr = np.linspace(0.0, t_days, steps)

        x_arr = np.zeros(steps)
        n_arr = np.zeros(steps)
        p_arr = np.zeros(steps)
        lipid_arr = np.zeros(steps)
        protein_arr = np.zeros(steps)
        carb_arr = np.zeros(steps)
        iavg_arr = np.zeros(steps)
        mu_arr = np.zeros(steps)

        # Set initials
        x = max(1e-4, initial_x)
        n = max(0.0, initial_n)
        p = max(0.0, initial_p)

        for i in range(steps):
            t = time_arr[i]
            i_avg = self.calculate_average_irradiance(i_surface, x)
            mu = self.calculate_growth_rate(i_avg, n, p)
            lip, prot, carb, _ = self.calculate_biochemical_composition(n)

            x_arr[i] = x
            n_arr[i] = n
            p_arr[i] = p
            lipid_arr[i] = lip
            protein_arr[i] = prot
            carb_arr[i] = carb
            iavg_arr[i] = i_avg
            mu_arr[i] = mu

            if i == steps - 1:
                break

            # RK4 Integration step
            def derivatives(curr_x: float, curr_n: float, curr_p: float):
                curr_iavg = self.calculate_average_irradiance(i_surface, curr_x)
                curr_mu = self.calculate_growth_rate(curr_iavg, curr_n, curr_p)
                # dX/dt = (mu - D) * X
                dx_dt = (curr_mu - dilution_rate) * curr_x
                # dN/dt = D*(Nin - N) - Y_N/X * mu * X * 1000 (mg/g conversion)
                # Note: yield_n_per_x is g N / g biomass = mg N / mg biomass
                # X is in g/L, so uptake is yield_n_per_x * X (g N / L / day) = 1000 * yield_n_per_x * X (mg/L/day)
                # only positive growth consumes nutrients:
                uptake_mu = max(0.0, curr_mu)
                dn_dt = dilution_rate * (inlet_n - curr_n) - (self.params.yield_n_per_x * 1000.0) * uptake_mu * curr_x
                dp_dt = dilution_rate * (inlet_p - curr_p) - (self.params.yield_p_per_x * 1000.0) * uptake_mu * curr_x
                return dx_dt, dn_dt, dp_dt

            k1_x, k1_n, k1_p = derivatives(x, n, p)
            k2_x, k2_n, k2_p = derivatives(
                x + 0.5 * dt_days * k1_x,
                n + 0.5 * dt_days * k1_n,
                p + 0.5 * dt_days * k1_p,
            )
            k3_x, k3_n, k3_p = derivatives(
                x + 0.5 * dt_days * k2_x,
                n + 0.5 * dt_days * k2_n,
                p + 0.5 * dt_days * k2_p,
            )
            k4_x, k4_n, k4_p = derivatives(
                x + dt_days * k3_x,
                n + dt_days * k3_n,
                p + dt_days * k3_p,
            )

            x = max(1e-5, x + (dt_days / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x))
            n = max(0.0, n + (dt_days / 6.0) * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n))
            p = max(0.0, p + (dt_days / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p))

        return PBRSimulationState(
            time_days=time_arr,
            biomass_g_l=x_arr,
            nitrogen_mg_l=n_arr,
            phosphorus_mg_l=p_arr,
            lipid_fraction=lipid_arr,
            protein_fraction=protein_arr,
            carb_fraction=carb_arr,
            average_irradiance=iavg_arr,
            growth_rate_mu=mu_arr,
        )


class HydrothermalLiquefactionReactor:
    """
    Continuous Hydrothermal Liquefaction (HTL) thermodynamic and yield calculation engine.
    Converts wet algal biomass into biocrude oil, aqueous coproducts, gas, and hydrochar.
    """

    def __init__(self, params: Optional[HTLReactorParams] = None):
        self.params = params or HTLReactorParams()

    def estimate_feedstock_hhv(
        self, lipid_frac: float, protein_frac: float, carb_frac: float, ash_frac: float
    ) -> float:
        """
        Calculates Higher Heating Value (HHV, MJ/kg) of raw dry biomass feedstock
        using component contributions:
        HHV_lipid ~ 39.5 MJ/kg, HHV_protein ~ 23.8 MJ/kg, HHV_carb ~ 17.3 MJ/kg, HHV_ash ~ 0.
        """
        hhv = (
            lipid_frac * 39.5
            + protein_frac * 23.8
            + carb_frac * 17.3
            + ash_frac * 0.0
        )
        return float(hhv)

    def calculate_biocrude_yield(
        self,
        lipid_frac: float,
        protein_frac: float,
        carb_frac: float,
        ash_frac: float,
    ) -> HTLYieldResult:
        """
        Calculates biocrude oil yield, HHV, elemental partition, and energy recovery.
        Includes temperature and residence time non-linear kinetic corrections.
        """
        # Temperature efficiency curve peaking around 340-350 C
        t_c = self.params.temperature_c
        if t_c < 250.0 or t_c > 450.0:
            raise ValueError(f"HTL temperature {t_c}°C outside valid operating range [250, 450].")

        t_factor = 1.0 - 0.00015 * ((t_c - 340.0) ** 2)
        t_factor = max(0.65, min(1.05, t_factor))

        # Residence time saturation: 1 - exp(-tau / 15 min)
        tau_factor = 1.0 - np.exp(-self.params.residence_time_min / 12.0)

        # Primary biocrude yield from macromolecules (dry ash-free basis scaled to dry total)
        daf_fraction = max(0.1, 1.0 - ash_frac)
        raw_oil_yield = (
            lipid_frac * self.params.conv_lipid
            + protein_frac * self.params.conv_protein
            + carb_frac * self.params.conv_carbohydrate
        ) * t_factor * tau_factor

        # Bound oil yield
        oil_yield = float(np.clip(raw_oil_yield, 0.05, 0.75))

        # Phase partitions (wt% of dry biomass)
        solid_char = float(np.clip(0.08 + 0.6 * ash_frac + 0.05 * (1.0 - tau_factor), 0.05, 0.35))
        gas_yield = float(np.clip(0.08 + 0.0004 * (t_c - 280.0), 0.05, 0.25))
        aqueous_yield = float(max(0.05, 1.0 - (oil_yield + solid_char + gas_yield)))

        # Renormalize to exact 100% mass balance
        total_mass = oil_yield + solid_char + gas_yield + aqueous_yield
        oil_yield_pct = (oil_yield / total_mass) * 100.0
        aqueous_pct = (aqueous_yield / total_mass) * 100.0
        gas_pct = (gas_yield / total_mass) * 100.0
        char_pct = (solid_char / total_mass) * 100.0

        # Elemental Biocrude composition estimation (%wt)
        # Driven by lipid vs protein vs carb contributions in product
        c_pct = float(np.clip(72.0 + 6.0 * lipid_frac - 4.0 * carb_frac, 65.0, 80.0))
        h_pct = float(np.clip(9.0 + 3.0 * lipid_frac - 1.5 * protein_frac, 7.5, 12.0))
        n_pct = float(np.clip(2.0 + 6.0 * protein_frac, 1.5, 7.5))
        o_pct = float(np.clip(100.0 - (c_pct + h_pct + n_pct + 0.5), 4.0, 18.0))

        # Higher Heating Value via Channiwala-Parikh formula:
        # HHV (MJ/kg) = 0.3491*C + 1.1783*H + 0.1005*S - 0.1034*O - 0.0151*N - 0.0211*Ash
        biocrude_hhv = float(
            0.3491 * c_pct
            + 1.1783 * h_pct
            + 0.1005 * 0.5
            - 0.1034 * o_pct
            - 0.0151 * n_pct
        )
        biocrude_hhv = max(28.0, min(42.0, biocrude_hhv))

        feedstock_hhv = self.estimate_feedstock_hhv(lipid_frac, protein_frac, carb_frac, ash_frac)

        # Energy Recovery: ER (%) = (Yield_oil * HHV_oil) / HHV_feedstock
        energy_recovery = float((oil_yield * biocrude_hhv / max(10.0, feedstock_hhv)) * 100.0)
        energy_recovery = min(95.0, energy_recovery)

        # Carbon Recovery (%)
        feedstock_c = 50.0 * (1.0 - ash_frac)  # Approximate dry ash-free algae C ~ 50%
        carbon_recovery = float((oil_yield * c_pct / max(10.0, feedstock_c)) * 100.0)
        carbon_recovery = min(90.0, carbon_recovery)

        return HTLYieldResult(
            biocrude_yield_wt_pct=oil_yield_pct,
            aqueous_yield_wt_pct=aqueous_pct,
            gas_yield_wt_pct=gas_pct,
            solid_char_yield_wt_pct=char_pct,
            biocrude_hhv_mj_kg=biocrude_hhv,
            feedstock_hhv_mj_kg=feedstock_hhv,
            energy_recovery_pct=energy_recovery,
            carbon_recovery_pct=carbon_recovery,
            elemental_biocrude_c=c_pct,
            elemental_biocrude_h=h_pct,
            elemental_biocrude_n=n_pct,
            elemental_biocrude_o=o_pct,
        )


def run_integrated_pbr_htl_pipeline(
    sim_days: float = 12.0,
    i_surface: float = 300.0,
    htl_temp_c: float = 345.0,
    htl_residence_min: float = 35.0,
) -> Tuple[PBRSimulationState, HTLYieldResult]:
    """
    End-to-end integration running photobioreactor biomass growth followed by
    continuous hydrothermal liquefaction of the harvested biomass.
    """
    pbr = MicroalgaePhotobioreactor()
    pbr_state = pbr.simulate(t_days=sim_days, i_surface=i_surface)

    # Harvest final day biomass composition
    final_lipid = float(pbr_state.lipid_fraction[-1])
    final_prot = float(pbr_state.protein_fraction[-1])
    final_carb = float(pbr_state.carb_fraction[-1])
    ash_frac = pbr.params.ash_fraction

    htl_params = HTLReactorParams(
        temperature_c=htl_temp_c,
        residence_time_min=htl_residence_min,
    )
    htl = HydrothermalLiquefactionReactor(params=htl_params)
    htl_res = htl.calculate_biocrude_yield(final_lipid, final_prot, final_carb, ash_frac)

    return pbr_state, htl_res
