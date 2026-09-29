from engine_12_crispr_causal_fault_localization import CRISPRFaultLocalizationEngine, MicroserviceSpan

def test_crispr_fault_localization():
    engine = CRISPRFaultLocalizationEngine(seed=42)
    spans = [
        MicroserviceSpan("s1", "auth-service", 45.0, 200, "TTTA_OAUTH_HEALTHY_OKAY"),
        MicroserviceSpan("s2", "db-service", 850.0, 504, "TTTC_DATABASE_LATENCY_SPIKE"),
        MicroserviceSpan("s3", "frontend", 900.0, 500, "TTTG_FORWARD_ERROR_DOWNSTREAM")
    ]
    res = engine.collateral_cleavage_root_cause(spans)
    assert res["cleaved_spans"] >= 2
    assert res["root_cause_detected"] == 1.0
