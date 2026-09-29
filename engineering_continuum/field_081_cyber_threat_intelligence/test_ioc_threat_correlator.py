from ioc_threat_correlator import IOCThreatCorrelator

def test_threat_correlator():
    c = IOCThreatCorrelator()
    c.register_ioc("ip1", "ipv4")
    c.register_ioc("h1", "hash")
    c.add_correlation("ip1", "h1")
    assert len(c.find_campaign_cluster("ip1")) == 2
