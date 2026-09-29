from adhd_cognitive_scaffold import ADHDCognitiveScaffold

def test_adhd_scaffold():
    scaff = ADHDCognitiveScaffold()
    scaff.add_task("Clean Room", 50, 10)
    scaff.add_task("Quick Email", 5, 20)
    seq = scaff.get_optimal_sequence()
    assert seq[0]["name"] == "Quick Email"
