from hbt_second_order_coherence import HBTCoherenceAnalyzer

def test_hbt_antibunching():
    g2 = HBTCoherenceAnalyzer.calculate_g2_zero(coincidences=10, singles_detector1=1000, singles_detector2=1000, time_window_sec=1e-5, total_time_sec=1.0)
    assert abs(g2 - 1.0) < 1e-4
    assert HBTCoherenceAnalyzer.is_single_photon_emitter(0.15) is True
