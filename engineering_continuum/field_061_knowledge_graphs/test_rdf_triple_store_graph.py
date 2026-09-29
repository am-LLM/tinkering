from rdf_triple_store_graph import RDFGraphStore

def test_rdf_knowledge_graph():
    kg = RDFGraphStore()
    kg.add_triple("Sensor_1", "monitors", "Reactor_Alpha")
    kg.add_triple("Sensor_2", "monitors", "Reactor_Alpha")
    kg.add_triple("Sensor_3", "monitors", "Pump_Beta")
    
    matches = kg.query_pattern(("?sensor", "monitors", "Reactor_Alpha"))
    assert len(matches) == 2
    sensors = {m["?sensor"] for m in matches}
    assert "Sensor_1" in sensors and "Sensor_2" in sensors
