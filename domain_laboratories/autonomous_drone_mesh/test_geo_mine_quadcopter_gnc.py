"""
Unit Tests for Subterranean GPS-Denied Quadcopter GNC & Gas Plume Tracking
"""

import pytest
import numpy as np
from geo_mine_quadcopter_gnc import (
    OctreeMap, BoundingBox3D, SubterraneanGasEnvironment, GasPlumeSource,
    GasPlumeChemotaxisTracker, MinePathPlanner, QuadcopterFlightController
)


def test_octree_spatial_mapping_and_collision():
    # Setup 30x30x10 meter subterranean boundary
    octree = OctreeMap(bounds_min=np.array([0, 0, 0]), bounds_max=np.array([30, 30, 10]), resolution=1.0)
    
    # Add a collapsed tunnel wall block at [10..12, 0..20, 0..10]
    octree.add_box_obstacle(min_pt=np.array([10, 0, 0]), max_pt=np.array([12, 20, 10]))
    
    # Point inside the wall must collide
    assert octree.is_collision(np.array([11.0, 10.0, 5.0])) is True
    # Point outside bounds must collide
    assert octree.is_collision(np.array([-1.0, 5.0, 5.0])) is True
    # Point in clear open shaft must be free
    assert octree.is_collision(np.array([5.0, 5.0, 5.0])) is False


def test_gas_plume_gradient_chemotaxis():
    # Gas leak source at [15.0, 15.0, 5.0]
    source_loc = np.array([15.0, 15.0, 5.0])
    env = SubterraneanGasEnvironment(sources=[GasPlumeSource(origin=source_loc, peak_concentration=500.0)], background_noise=0.0)
    tracker = GasPlumeChemotaxisTracker(sample_baseline_radius=0.5)
    
    # Test drone at [10.0, 15.0, 5.0] (5 meters west of leak)
    drone_pos = np.array([10.0, 15.0, 5.0])
    grad = tracker.estimate_gradient_at(drone_pos, env)
    
    # Gradient in X must be positive (pointing towards X=15)
    assert grad[0] > 0.0, f"Gradient X should be positive, got {grad[0]}"
    # Gradient in Y and Z should be close to 0
    assert np.isclose(grad[1], 0.0, atol=1e-3)
    assert np.isclose(grad[2], 0.0, atol=1e-3)
    
    # Step along gradient for 5 iterations: distance to leak should decrease
    curr_pos = drone_pos.copy()
    initial_dist = np.linalg.norm(curr_pos - source_loc)
    
    for _ in range(5):
        g = tracker.estimate_gradient_at(curr_pos, env)
        step_dir = g / (np.linalg.norm(g) + 1e-6)
        curr_pos += step_dir * 0.8
        
    final_dist = np.linalg.norm(curr_pos - source_loc)
    assert final_dist < initial_dist, f"Chemotaxis failed to ascend plume: {initial_dist} -> {final_dist}"


def test_3d_mine_shaft_a_star_path_planning():
    octree = OctreeMap(bounds_min=np.array([0, 0, 0]), bounds_max=np.array([20, 20, 10]), resolution=1.0)
    
    # Add an obstacle barrier between x=8 and x=10 with a narrow passage at y=15..17
    # Wall from y=0 to y=14
    octree.add_box_obstacle(min_pt=np.array([9, 0, 0]), max_pt=np.array([10, 14, 10]))
    
    planner = MinePathPlanner(octree, step_size=1.0, safety_margin=0.2)
    start = np.array([2.0, 5.0, 3.0])
    goal = np.array([16.0, 5.0, 3.0])
    
    path = planner.plan_path(start, goal)
    assert path is not None, "Failed to find 3D path through shaft"
    assert len(path) >= 2
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)
    
    # Verify no waypoint in path collides with obstacle
    for wp in path:
        assert octree.is_collision(wp, margin=0.1) is False


def test_quadcopter_6dof_flight_controller():
    controller = QuadcopterFlightController(mass=1.2)
    controller.state.p = np.array([0.0, 0.0, 0.0])
    controller.state.v = np.array([0.0, 0.0, 0.0])
    
    target_waypoint = np.array([5.0, 5.0, 4.0])
    
    # Simulate flight for 100 steps
    dt = 0.05
    for _ in range(100):
        controller.step_simulation(target_waypoint, dt=dt)
        
    # Drone should have converged close to target
    dist = np.linalg.norm(controller.state.p - target_waypoint)
    assert dist < 0.5, f"Quadcopter failed to reach waypoint: final pos = {controller.state.p}, dist = {dist}"
    assert np.linalg.norm(controller.state.v) < controller.v_max + 1e-3
