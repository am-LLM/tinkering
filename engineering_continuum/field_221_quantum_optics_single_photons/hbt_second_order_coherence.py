"""Course 221: Hanbury Brown-Twiss g^(2)(0) Photon Antibunching Coherence Analyzer"""
class HBTCoherenceAnalyzer:
    @staticmethod
    def calculate_g2_zero(coincidences: int, singles_detector1: int, singles_detector2: int, time_window_sec: float, total_time_sec: float) -> float:
        expected_accidental = (singles_detector1 * singles_detector2 * time_window_sec) / total_time_sec
        if expected_accidental <= 0:
            return 0.0
        return float(coincidences / expected_accidental)

    @classmethod
    def is_single_photon_emitter(cls, g2_zero: float) -> bool:
        # Quantum threshold for single photon state: g^(2)(0) < 0.5
        return g2_zero < 0.5
