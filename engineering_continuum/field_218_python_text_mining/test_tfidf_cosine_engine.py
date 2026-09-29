from tfidf_cosine_engine import TFIDFCosineEngine

def test_tfidf_cosine():
    sim = TFIDFCosineEngine.compute_tfidf_and_cosine("machine learning algorithms", "machine learning algorithms")
    assert abs(sim - 1.0) < 1e-4
    sim_diff = TFIDFCosineEngine.compute_tfidf_and_cosine("quantum physics", "cooking recipes")
    assert sim_diff == 0.0
