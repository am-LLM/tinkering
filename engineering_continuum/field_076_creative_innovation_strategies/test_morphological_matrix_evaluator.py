from morphological_matrix_evaluator import MorphologicalMatrix

def test_morphological_scoring():
    mm = MorphologicalMatrix()
    mm.add_parameter("P1", ["A", "B"])
    mm.add_parameter("P2", ["C", "D"])
    assert mm.total_combinations() == 4
