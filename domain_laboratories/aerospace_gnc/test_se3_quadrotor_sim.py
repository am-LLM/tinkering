"""
Unit Tests for SE(3) Quadrotor Simulator, Differential Flatness & Minimum Snap Trajectory
"""

import pytest
import numpy as np
from se3_quadrotor_sim import (
    SE3QuadrotorSimulator, QuadrotorConfig, QuadrotorState,
    DifferentialFlatness, MinimumSnapTrajectoryGenerator,
    SE3GeometricController, skew, vee, rot_matrix_to_quat
)


def test_skew_and_vee_operators():
    v = np.array([1.5, -2.3, 4.1])
    S = skew(v)
    assert np.allclose(S, -S.T)
    v_rec = vee(S)
    assert np.allclose(v, v_rec)


def test_rot_matrix_to_quat_branches():
    R_id = np.eye(3)
    q_id = rot_matrix_to_quat(R_id)
    assert np.allclose(q_id, np.array([1.0, 0.0, 0.0, 0.0]))
    
    R_x = np.diag([1.0, -1.0, -1.0])
    q_x = rot_matrix_to_quat(R_x)
    assert np.isclose(np.abs(q_x[1]), 1.0)
    
    R_y = np.diag([-1.0, 1.0, -1.0])
    q_y = rot_matrix_to_quat(R_y)
    assert np.isclose(np.abs(q_y[2]), 1.0)
    
    R_z = np.diag([-1.0, -1.0, 1.0])
    q_z = rot_matrix_to_quat(R_z)
    assert np.isclose(np.abs(q_z[3]), 1.0)


def test_quadrotor_mixer_and_wrench():
    sim = SE3QuadrotorSimulator()
    hover_f_each = (sim.config.mass * sim.config.gravity) / 4.0
    motors = np.array([hover_f_each, hover_f_each, hover_f_each, hover_f_each])
    
    f_total, tau = sim.motor_thrusts_to_wrench(motors)
    assert np.isclose(f_total, sim.config.mass * sim.config.gravity)
    assert np.allclose(tau, np.zeros(3), atol=1e-6)
    
    motors_recomputed = sim.wrench_to_motor_thrusts(f_total, tau)
    assert np.allclose(motors, motors_recomputed)
    
    state = sim.step_motors(motors, dt=0.01)
    assert isinstance(state, QuadrotorState)


def test_quadrotor_hover_rk4_simulation():
    sim = SE3QuadrotorSimulator()
    sim.set_state(p=[0, 0, 10], v=[0, 0, 0], R=np.eye(3), omega=[0, 0, 0])
    
    hover_f = sim.config.mass * sim.config.gravity
    dt = 0.01
    
    for _ in range(100):
        sim.step_wrench(hover_f, np.zeros(3), dt)
        
    assert np.allclose(sim.state.p, np.array([0.0, 0.0, 10.0]), atol=1e-3)
    assert np.allclose(sim.state.v, np.zeros(3), atol=1e-3)
    assert len(sim.state.quaternion) == 4

    # Test reflection matrix SVD determinant correction branch
    sim.state.R = np.diag([1.0, 1.0, -1.0])
    sim.step_wrench(hover_f, np.zeros(3), dt)
    assert np.isclose(np.linalg.det(sim.state.R), 1.0)


def test_differential_flatness():
    flat = DifferentialFlatness(mass=1.0, gravity=9.80665)
    
    res_hover = flat.compute_state_and_input(
        p=[0, 0, 5], v=[0, 0, 0], a=[0, 0, 0],
        j=[0, 0, 0], s=[0, 0, 0],
        yaw=0.0, yaw_dot=0.0
    )
    assert np.isclose(res_hover["thrust"], 9.80665)
    assert np.allclose(res_hover["R"], np.eye(3))
    assert np.allclose(res_hover["omega"], np.zeros(3))
    
    res_acc = flat.compute_state_and_input(
        p=[0, 0, 5], v=[1, 0, 0], a=[2.0, 0, 0],
        j=[0.1, 0, 0], s=[0, 0, 0],
        yaw=0.0, yaw_dot=0.0
    )
    assert res_acc["thrust"] > 9.80665
    assert res_acc["R"][0, 2] > 0.0

    res_sing = flat.compute_state_and_input(
        p=[0, 0, 0], v=[0, 0, 0], a=[0, 0, -9.80665],
        j=[0, 0, 0], s=[0, 0, 0],
        yaw=0.0, yaw_dot=0.0
    )
    assert res_sing["thrust"] >= 0.0


def test_minimum_snap_trajectory_generator():
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 3.0],
        [5.0, 4.0, 2.0]
    ])
    time_allocations = np.array([2.0, 2.0])
    
    gen = MinimumSnapTrajectoryGenerator(waypoints, time_allocations)
    
    s0 = gen.evaluate(0.0)
    assert np.allclose(s0["p"], waypoints[0], atol=1e-4)
    assert np.allclose(s0["v"], np.zeros(3), atol=1e-4)
    
    s_mid = gen.evaluate(2.0)
    assert np.allclose(s_mid["p"], waypoints[1], atol=1e-3)
    
    s_end = gen.evaluate(4.0)
    assert np.allclose(s_end["p"], waypoints[2], atol=1e-4)
    assert np.allclose(s_end["v"], np.zeros(3), atol=1e-4)

    s_past = gen.evaluate(10.0)
    assert np.allclose(s_past["p"], waypoints[2], atol=1e-4)


def test_se3_geometric_controller_tracking():
    config = QuadrotorConfig(mass=1.0)
    controller = SE3GeometricController(config=config, kp=4.0, kv=3.0, kr=30.0, kw=8.0)
    sim = SE3QuadrotorSimulator(config=config)
    
    sim.set_state(p=[0.5, -0.5, 1.0], v=[0.0, 0.0, 0.0], R=np.eye(3), omega=[0.0, 0.0, 0.0])
    
    target_p = np.array([0.0, 0.0, 2.0])
    target_v = np.zeros(3)
    target_a = np.zeros(3)
    
    dt = 0.01
    for step in range(400):
        f_thrust, tau = controller.compute_control(sim.state, target_p, target_v, target_a, des_yaw=0.0)
        sim.step_wrench(f_thrust, tau, dt)
        
    assert np.linalg.norm(sim.state.p - target_p) < 0.05
    assert np.linalg.norm(sim.state.v) < 0.1

    f_zero, tau_zero = controller.compute_control(
        sim.state, des_p=sim.state.p, des_v=sim.state.v, des_a=np.array([0.0, 0.0, -sim.config.gravity]), des_yaw=0.0
    )
    assert f_zero >= 0.0
