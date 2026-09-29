from occupancy_grid_slam import OccupancyGridSLAM

def test_occupancy_grid():
    slam = OccupancyGridSLAM()
    slam.update_ray(0, 0, 5, 5)
    probs = slam.get_probability_grid()
    assert probs[5, 5] > 0.5
