from schelling_segregation_agent import SchellingGrid

def test_schelling_segregation():
    grid = SchellingGrid(size=10, similarity_threshold=0.4)
    initial_unhappy = grid.step()
    for _ in range(10):
        grid.step()
    final_unhappy = grid.step()
    assert final_unhappy <= initial_unhappy
