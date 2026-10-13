import pytest
import numpy as np
from frontier_hybrids.engine_73_neuromorphic_active_inference_homeostasis import NeuromorphicActiveInferenceEngine

def test_engine_73_initialization():
    engine = NeuromorphicActiveInferenceEngine()
    assert engine.dbs_base_freq_hz == 130.0
    assert engine.target_beta_power == 0.15

def test_engine_73_optimal_homeostasis():
    engine = NeuromorphicActiveInferenceEngine()
    
    # Healthy nominal biological and hardware telemetry
    beta_power = 0.16
    battery_soc = 90.0
    temp_c = 42.0
    ram = 0.30
    
    res = engine.execute_homeostasis_step(beta_power, battery_soc, temp_c, ram)
    assert res["splenic_active_gear"] == "FULL_SPECTRUM_PARALLEL"
    assert res["compute_bandwidth_scale"] == 1.00
    assert res["variational_free_energy"] < 0.05
    assert res["dbs_controller"]["mode"] == "ENERGY_SAVING_IDLE"
    assert res["symbiotic_state"] == "OPTIMAL_HOMEOSTASIS"

def test_engine_73_critical_starvation_and_pathological_burst():
    engine = NeuromorphicActiveInferenceEngine()
    
    # Pathological beta burst + critical battery/thermal stress
    beta_power = 0.65 # Severe Parkinsonian/tremor synchronization
    battery_soc = 8.0 # Battery nearly dead
    temp_c = 88.0    # Thermal runaway
    ram = 0.98       # Out-of-memory imminent
    
    res = engine.execute_homeostasis_step(beta_power, battery_soc, temp_c, ram)
    assert res["splenic_active_gear"] == "ASPHYXIC_ATOMIC_LUT"
    assert res["compute_bandwidth_scale"] == 0.10
    assert res["variational_free_energy"] > 0.35
    assert res["dbs_controller"]["mode"] == "PROPORTIONAL_CLOSED_LOOP"
    assert res["dbs_controller"]["frequency_hz"] > 160.0 # Elevated frequency to break sync
    assert res["symbiotic_state"] == "CRITICAL_STARVATION_RECOVERY"
    assert res["cgp_adaptation"]["cgp_active_dag_nodes"] <= 4 # Culled DAG nodes to save compute
