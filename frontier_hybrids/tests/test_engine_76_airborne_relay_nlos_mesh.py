import pytest
import numpy as np
from frontier_hybrids.engine_76_airborne_relay_nlos_mesh import AirborneRelayMeshEngine

def test_engine_76_initialization():
    engine = AirborneRelayMeshEngine(carrier_freq_mhz=868.0)
    assert engine.freq_mhz == 868.0
    assert engine.tx_power == 27.0
    assert len(engine.buffered_packets) == 0

def test_engine_76_airborne_relay_fallback():
    engine = AirborneRelayMeshEngine()
    
    ground_pos = np.array([0.0, 0.0, 2.0])
    relay_pos = np.array([1000.0, 0.0, 400.0]) # High hover at 400m
    penetrator_pos = np.array([2000.0, 0.0, 1.0]) # Behind deep hill/ridge
    
    # Direct ground link is occluded by hill
    res = engine.route_two_tier_telemetry(
        ground_pos, relay_pos, penetrator_pos, penetrator_occluded=True, payload_packet={"id": 1}
    )
    
    assert res["direct_link_margin_db"] < 0.0 # Direct failed
    assert res["routing_mode"] == "AIRBORNE_RELAY_MOTHERSHIP"
    assert res["packet_delivered"] is True

def test_engine_76_store_and_forward_buffer():
    engine = AirborneRelayMeshEngine()
    
    ground_pos = np.array([0.0, 0.0, 2.0])
    # Relay absent or down
    relay_pos = np.array([0.0, 0.0, 0.0])
    penetrator_pos = np.array([8000.0, 0.0, 1.0])
    
    # Complete blackout
    res = engine.route_two_tier_telemetry(
        ground_pos, relay_pos, penetrator_pos, penetrator_occluded=True, payload_packet={"sensor": "telemetry_01"}
    )
    
    assert res["packet_delivered"] is False
    assert res["routing_mode"] == "STORE_AND_FORWARD_BUFFERED"
    assert res["buffered_packets_count"] == 1
