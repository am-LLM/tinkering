"""
Comprehensive Unit Test Suite for Bio-Extremophile & Critical Medical Systems.

Covers:
1. synthetic_crispr_gillespie_circuits.py
2. algae_photobioreactor_hydrothermal.py
3. evidence_based_pharmacognosy_engine.py
4. trauma_damage_control_triage_sim.py
"""

import math
import numpy as np
import pytest

from synthetic_crispr_gillespie_circuits import (
    CircuitType,
    GillespieCircuitEngine,
    GillespieSimulationResult,
    Reaction,
    RepressilatorParams,
    ToggleSwitchParams,
    build_crispr_repressilator,
    build_crispr_toggle_switch,
    simulate_repressilator,
    simulate_toggle_switch,
)

from algae_photobioreactor_hydrothermal import (
    HTLReactorParams,
    HTLYieldResult,
    HydrothermalLiquefactionReactor,
    MicroalgaePhotobioreactor,
    PBRSimulationState,
    PhotobioreactorParams,
    run_integrated_pbr_htl_pipeline,
)

from evidence_based_pharmacognosy_engine import (
    EnzymeKineticsModel,
    InhibitionType,
    PharmacognosyRepository,
    PharmacognosySynergyEngine,
    SecondaryMetabolite,
)

from trauma_damage_control_triage_sim import (
    PatientBaseline,
    PhysiologicalSnapshot,
    ResuscitationEvent,
    ResuscitationFluidType,
    TraumaHemodynamicSimulator,
    TriageCategory,
)


# ============================================================================
# 1. SYNTHETIC CRISPR GILLESPIE CIRCUITS TESTS
# ============================================================================

def test_gillespie_engine_empty_and_errors():
    engine = GillespieCircuitEngine()
    with pytest.raises(RuntimeError):
        engine.run_simulation(t_max=10.0)

    with pytest.raises(ValueError):
        engine.set_initial_state({"A": -5})

    with pytest.raises(ValueError):
        engine.run_simulation(t_max=-1.0)


def test_gillespie_simple_decay_reaction():
    engine = GillespieCircuitEngine(rng_seed=123)
    engine.set_initial_state({"X": 100})
    engine.add_reaction(
        Reaction(
            name="Decay_X",
            reactants={"X": 1},
            products={},
            propensity_fn=lambda st: 0.1 * st.get("X", 0),
        )
    )
    result = engine.run_simulation(t_max=50.0, record_interval=5.0)
    assert isinstance(result, GillespieSimulationResult)
    assert len(result.time) > 1
    assert result.state_trajectories["X"][-1] < 100
    assert result.reaction_counts["Decay_X"] > 0


def test_gillespie_engine_clear():
    engine = GillespieCircuitEngine()
    engine.set_initial_state({"X": 10})
    engine.add_reaction(Reaction("R1", {}, {}, lambda st: 1.0))
    assert len(engine.reactions) == 1
    assert len(engine.species_initial) == 1
    engine.clear()
    assert len(engine.reactions) == 0
    assert len(engine.species_initial) == 0


def test_crispr_toggle_switch_simulation():
    params = ToggleSwitchParams(k_tx_u=15.0, k_tx_v=15.0, k_tl_u=10.0, k_tl_v=10.0)
    init_u_high = {"M_u": 15, "P_u": 150, "M_v": 0, "P_v": 0}
    res = simulate_toggle_switch(t_max=80.0, record_interval=2.0, params=params, initial_counts=init_u_high, rng_seed=42)

    assert res.circuit_type == CircuitType.TOGGLE_SWITCH
    final_pu = res.get_species_final("P_u")
    final_pv = res.get_species_final("P_v")
    assert final_pu > final_pv
    bistability_ratio = res.calculate_toggle_bistability_ratio()
    assert bistability_ratio > 0.0

    with pytest.raises(KeyError):
        res.get_species_final("NonExistentSpecies")


def test_crispr_repressilator_simulation():
    params = RepressilatorParams(k_tx=20.0, k_tl=12.0, n_hill=3.0, k_rep=20.0)
    res = simulate_repressilator(t_max=150.0, record_interval=1.0, params=params, rng_seed=101)

    assert res.circuit_type == CircuitType.REPRESSILATOR
    assert "P_x" in res.state_trajectories
    assert "P_y" in res.state_trajectories
    assert "P_z" in res.state_trajectories
    assert len(res.time) == len(res.state_trajectories["P_x"])

    is_osc, period, amp = res.detect_repressilator_oscillations(species="P_x", min_peaks=1)
    assert isinstance(is_osc, (bool, np.bool_))
    assert period >= 0.0
    assert amp >= 0.0

    with pytest.raises(KeyError):
        res.detect_repressilator_oscillations(species="Unknown")


# ============================================================================
# 2. ALGAE PHOTOBIOREACTOR & HTL TESTS
# ============================================================================

def test_pbr_average_irradiance_attenuation():
    pbr = MicroalgaePhotobioreactor()
    # Zero surface light
    assert pbr.calculate_average_irradiance(0.0, 1.0) == 0.0
    # Clear water (minimal biomass)
    i_avg_clear = pbr.calculate_average_irradiance(500.0, 0.0001)
    assert math.isclose(i_avg_clear, 500.0, rel_tol=1e-2)
    # Dense biomass light extinction
    i_avg_dense = pbr.calculate_average_irradiance(500.0, 10.0)
    assert i_avg_dense < 500.0
    assert i_avg_dense > 0.0


def test_pbr_growth_kinetics():
    pbr = MicroalgaePhotobioreactor()
    # Dark condition -> decay
    mu_dark = pbr.calculate_growth_rate(0.0, 50.0, 5.0)
    assert mu_dark < 0.0

    # Optimal light + nutrient replete
    mu_opt = pbr.calculate_growth_rate(pbr.params.i_opt, 50.0, 5.0)
    assert mu_opt > 0.5

    # Nitrogen starved
    mu_n_starved = pbr.calculate_growth_rate(pbr.params.i_opt, 0.001, 5.0)
    assert mu_n_starved < mu_opt


def test_pbr_lipid_accumulation_under_starvation():
    pbr = MicroalgaePhotobioreactor()
    lipid_replete, prot_replete, carb_replete, ash = pbr.calculate_biochemical_composition(nitrogen_n=100.0)
    lipid_starved, prot_starved, carb_starved, _ = pbr.calculate_biochemical_composition(nitrogen_n=0.01)

    assert lipid_starved > lipid_replete
    assert prot_starved < prot_replete
    assert math.isclose(lipid_replete + prot_replete + carb_replete + ash, 1.0, abs_tol=1e-3)
    assert math.isclose(lipid_starved + prot_starved + carb_starved + ash, 1.0, abs_tol=1e-3)


def test_pbr_dynamic_simulation_batch():
    pbr = MicroalgaePhotobioreactor()
    state = pbr.simulate(t_days=7.0, dt_days=0.1, initial_x=0.2, initial_n=50.0, initial_p=6.0)
    assert isinstance(state, PBRSimulationState)
    assert len(state.time_days) == len(state.biomass_g_l)
    assert state.biomass_g_l[-1] > state.biomass_g_l[0]  # Growth occurred
    assert state.nitrogen_mg_l[-1] < state.nitrogen_mg_l[0]  # Nutrients consumed


def test_pbr_continuous_chemostat():
    pbr = MicroalgaePhotobioreactor()
    state = pbr.simulate(
        t_days=10.0,
        dt_days=0.1,
        initial_x=0.5,
        initial_n=30.0,
        initial_p=4.0,
        dilution_rate=0.2,
        inlet_n=50.0,
        inlet_p=6.0,
    )
    assert state.biomass_g_l[-1] > 0.1
    assert state.nitrogen_mg_l[-1] > 0.0


def test_pbr_invalid_times():
    pbr = MicroalgaePhotobioreactor()
    with pytest.raises(ValueError):
        pbr.simulate(t_days=-5.0)


def test_htl_biocrude_yield_and_energy_recovery():
    htl = HydrothermalLiquefactionReactor(HTLReactorParams(temperature_c=350.0, residence_time_min=30.0))
    res = htl.calculate_biocrude_yield(lipid_frac=0.35, protein_frac=0.30, carb_frac=0.25, ash_frac=0.10)

    assert isinstance(res, HTLYieldResult)
    # Mass balance closure
    total_yield = (
        res.biocrude_yield_wt_pct
        + res.aqueous_yield_wt_pct
        + res.gas_yield_wt_pct
        + res.solid_char_yield_wt_pct
    )
    assert math.isclose(total_yield, 100.0, abs_tol=0.5)

    # HHV and ER checks
    assert 28.0 <= res.biocrude_hhv_mj_kg <= 42.0
    assert 40.0 <= res.energy_recovery_pct <= 95.0
    assert 40.0 <= res.carbon_recovery_pct <= 90.0


def test_htl_temperature_out_of_bounds():
    htl_cold = HydrothermalLiquefactionReactor(HTLReactorParams(temperature_c=150.0))
    with pytest.raises(ValueError):
        htl_cold.calculate_biocrude_yield(0.2, 0.4, 0.3, 0.1)


def test_integrated_pbr_htl_pipeline():
    state, htl_res = run_integrated_pbr_htl_pipeline(sim_days=5.0, i_surface=250.0)
    assert len(state.time_days) > 10
    assert htl_res.biocrude_yield_wt_pct > 10.0


# ============================================================================
# 3. EVIDENCE-BASED PHARMACOGNOSY ENGINE TESTS
# ============================================================================

def test_pharmacognosy_database_integrity():
    db = PharmacognosyRepository.get_standard_database()
    assert "salicin" in db
    assert "artemisinin" in db
    assert "berberine" in db
    assert "medical_grade_honey_mgo" in db

    for key, met in db.items():
        assert isinstance(met, SecondaryMetabolite)
        assert met.molecular_weight > 0
        assert met.reported_ic50_um > 0
        passes_lip, violations = met.passes_lipinski_rule_of_five()
        assert isinstance(passes_lip, bool)
        ki = met.calculate_dissociation_constant_ki()
        assert ki > 0.0


def test_enzyme_inhibition_modes():
    vmax = 100.0
    km = 10.0
    ki = 5.0
    s = 10.0
    i_conc = 5.0

    # Uninhibited
    model_none = EnzymeKineticsModel(v_max=vmax, k_m=km)
    v_uninhibited = model_none.velocity(s, 0.0)
    assert math.isclose(v_uninhibited, 50.0, rel_tol=1e-3)
    assert model_none.velocity(0.0, 0.0) == 0.0

    # Competitive
    model_comp = EnzymeKineticsModel.create(vmax, km, InhibitionType.COMPETITIVE, ki)
    v_comp = model_comp.velocity(s, i_conc)
    assert v_comp < v_uninhibited
    # In competitive inhibition, saturating substrate overcomes inhibitor
    v_comp_sat = model_comp.velocity(10000.0, i_conc)
    assert math.isclose(v_comp_sat, vmax, rel_tol=1e-2)

    # Uncompetitive
    model_uncomp = EnzymeKineticsModel.create(vmax, km, InhibitionType.UNCOMPETITIVE, ki)
    v_uncomp = model_uncomp.velocity(s, i_conc)
    assert v_uncomp < v_uninhibited

    # Non-Competitive
    model_noncomp = EnzymeKineticsModel.create(vmax, km, InhibitionType.NONCOMPETITIVE, ki)
    v_noncomp = model_noncomp.velocity(s, i_conc)
    assert v_noncomp < v_uninhibited

    # Mixed
    model_mixed = EnzymeKineticsModel.create(vmax, km, InhibitionType.MIXED, ki, alpha_mixed=3.0)
    v_mixed = model_mixed.velocity(s, i_conc)
    assert v_mixed < v_uninhibited


def test_kinetic_transformations():
    model = EnzymeKineticsModel(v_max=80.0, k_m=12.0)
    s_points = np.array([2.0, 5.0, 10.0, 25.0, 50.0])

    inv_s, inv_v = model.lineweaver_burk_coords(s_points)
    assert len(inv_s) == len(s_points)
    assert np.all(inv_s > 0)
    assert np.all(inv_v > 0)

    with pytest.raises(ValueError):
        model.lineweaver_burk_coords(np.array([0.0, 10.0]))

    s_h, s_over_v = model.hanes_woolf_coords(s_points)
    assert len(s_h) == len(s_points)

    v_over_s, v_eh = model.eadie_hofstee_coords(s_points)
    assert len(v_over_s) == len(s_points)


def test_chou_talalay_synergy_engine():
    # Synergistic dose combo
    ci_syn = PharmacognosySynergyEngine.calculate_combination_index(
        dose_1=2.0, dose_2=1.0, ic50_1=10.0, ic50_2=10.0, effect_fa=0.5
    )
    assert ci_syn < 1.0
    label_syn = PharmacognosySynergyEngine.classify_synergy(ci_syn)
    assert "Synerg" in label_syn

    # Antagonistic combo
    ci_ant = PharmacognosySynergyEngine.calculate_combination_index(
        dose_1=15.0, dose_2=15.0, ic50_1=10.0, ic50_2=10.0, effect_fa=0.5
    )
    assert ci_ant > 1.0
    label_ant = PharmacognosySynergyEngine.classify_synergy(ci_ant)
    assert "Antagonism" in label_ant

    # Test classifications
    assert PharmacognosySynergyEngine.classify_synergy(0.2) == "Strong Synergism"
    assert PharmacognosySynergyEngine.classify_synergy(0.8) == "Moderate Synergism"
    assert PharmacognosySynergyEngine.classify_synergy(1.0) == "Nearly Additive"
    assert PharmacognosySynergyEngine.classify_synergy(1.3) == "Slight Antagonism"
    assert PharmacognosySynergyEngine.classify_synergy(4.0) == "Strong Antagonism"

    with pytest.raises(ValueError):
        PharmacognosySynergyEngine.calculate_combination_index(1.0, 1.0, 10.0, 10.0, effect_fa=1.5)


# ============================================================================
# 4. TRAUMA DAMAGE CONTROL TRIAGE SIMULATOR TESTS
# ============================================================================

def test_trauma_simulator_invalid_inputs():
    sim = TraumaHemodynamicSimulator()
    with pytest.raises(ValueError):
        sim.simulate(duration_min=-10.0)


def test_trauma_unresuscitated_shock_collapse():
    sim = TraumaHemodynamicSimulator()
    # Severe unresuscitated bleed (150 mL/min) for 30 minutes
    timeline = sim.simulate(duration_min=30.0, dt_min=0.2, initial_bleed_rate_ml_min=150.0, events=[])

    assert len(timeline) > 50
    initial_snap = timeline[0]
    final_snap = timeline[-1]

    # Progressive hypovolemia
    assert final_snap.blood_volume_ml < initial_snap.blood_volume_ml
    assert final_snap.sbp_mmhg < initial_snap.sbp_mmhg
    assert final_snap.shock_index > initial_snap.shock_index
    assert final_snap.lactate_mmol_l > initial_snap.lactate_mmol_l
    assert final_snap.lethal_triad_severity_index > initial_snap.lethal_triad_severity_index
    assert final_snap.survival_probability < initial_snap.survival_probability


def test_trauma_damage_control_resuscitation_protocol():
    sim = TraumaHemodynamicSimulator()
    # Hemorrhage managed with 1:1:1 Blood, TXA, and surgical hemostasis at t=10 min
    events = [
        ResuscitationEvent(
            time_min=10.0,
            fluid_type=ResuscitationFluidType.BALANCED_1_1_1,
            volume_ml=1500.0,
            fluid_temperature_c=37.0,
            give_txa=True,
            apply_surgical_hemostasis=True,
            hemostasis_efficacy=0.95,
        ),
    ]
    timeline = sim.simulate(duration_min=45.0, dt_min=0.2, initial_bleed_rate_ml_min=120.0, events=events)

    snap_at_9min = [s for s in timeline if s.time_min <= 9.0][-1]
    snap_at_40min = [s for s in timeline if s.time_min <= 40.0][-1]

    # Hemostatic recovery
    assert snap_at_40min.map_mmhg > snap_at_9min.map_mmhg
    assert snap_at_40min.shock_index < snap_at_9min.shock_index
    assert snap_at_40min.survival_probability > snap_at_9min.survival_probability
    assert snap_at_40min.triage_tag in [TriageCategory.GREEN, TriageCategory.YELLOW]


def test_crystalloid_hemodilution_effect():
    sim = TraumaHemodynamicSimulator()
    # Large cold crystalloid bolus causing hemodilution and hypothermia
    events = [
        ResuscitationEvent(
            time_min=5.0,
            fluid_type=ResuscitationFluidType.CRYSTALLOID_SALINE,
            volume_ml=3000.0,
            fluid_temperature_c=20.0,  # Un-warmed cold saline
            give_txa=False,
            apply_surgical_hemostasis=False,
        )
    ]
    timeline = sim.simulate(duration_min=20.0, dt_min=0.2, initial_bleed_rate_ml_min=80.0, events=events)
    snap_post_fluid = [s for s in timeline if s.time_min <= 15.0][-1]

    # Hemodilution drops Hb and worsens INR
    assert snap_post_fluid.hemoglobin_g_dl < 12.0
    assert snap_post_fluid.temperature_c < 36.5
    assert snap_post_fluid.inr > 1.2


def test_prbc_only_transfusion():
    sim = TraumaHemodynamicSimulator()
    events = [
        ResuscitationEvent(
            time_min=5.0,
            fluid_type=ResuscitationFluidType.PRBC_ONLY,
            volume_ml=1500.0,
            fluid_temperature_c=37.0,
        )
    ]
    timeline = sim.simulate(duration_min=20.0, dt_min=0.2, initial_bleed_rate_ml_min=60.0, events=events)
    snap_post = [s for s in timeline if s.time_min <= 15.0][-1]
    # Hb elevated from packed red cells
    assert snap_post.hemoglobin_g_dl > 14.0
    # Clotting factors diluted without plasma
    assert snap_post.fibrinogen_mg_dl < 300.0
