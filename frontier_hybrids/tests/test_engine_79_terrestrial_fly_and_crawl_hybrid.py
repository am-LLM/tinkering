import pytest
import numpy as np
from frontier_hybrids.engine_79_terrestrial_fly_and_crawl_hybrid import TerrestrialFlyAndCrawlEngine

def test_engine_79_initialization():
    engine = TerrestrialFlyAndCrawlEngine()
    assert engine.mode == "TERRESTRIAL_ROLLING"
    assert engine.hover_power_watts == 75.0

def test_engine_79_ground_rolling_power_savings():
    engine = TerrestrialFlyAndCrawlEngine()
    
    # Smooth flat surface at 2 m/s
    res = engine.arbitrate_locomotion_step(target_speed_m_s=2.0, surface_obstacle_height_m=0.0, terrain_slope_deg=0.0)
    assert res["locomotion_mode"] == "TERRESTRIAL_ROLLING"
    assert res["power_draw_watts"] < 15.0 # Dramatic power drop
    assert res["energy_savings_pct"] > 80.0
    assert res["endurance_multiplier"] > 5.0
    assert res["rotor_throttle_pct"] == 12.0

def test_engine_79_aerial_obstacle_hop():
    engine = TerrestrialFlyAndCrawlEngine()
    
    # 20cm obstacle (brick/curb/rubble) forces vertical hop
    res = engine.arbitrate_locomotion_step(target_speed_m_s=2.0, surface_obstacle_height_m=0.20, terrain_slope_deg=0.0)
    assert res["locomotion_mode"] == "AERIAL_OBSTACLE_HOP"
    assert res["power_draw_watts"] == 75.0
    assert res["rotor_throttle_pct"] == 65.0
