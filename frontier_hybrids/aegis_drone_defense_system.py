"""
AEGIS-C-UAS: Autonomous Multi-Medium Air/Sea Drone Defense & Interception Engine
Author: Ali Malik (@am-LLM)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class AcousticSensorArrayNetwork:
    """
    Distributed Passive Acoustic Triangulation Network (Zvook/Sky Fortress Model).
    Triangulates low-flying composite UAS and USVs from distributed ground/buoy microphones.
    """
    def __init__(self, stations: np.ndarray, speed_of_sound: float = 343.0):
        self.stations = np.array(stations, dtype=float)
        self.c = float(speed_of_sound)
        self.num_stations = len(stations)

    def triangulate_threat_tdoa(self, time_of_arrivals: np.ndarray, init_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Gauss-Newton Non-Linear Least Squares TDoA Multilateration.
        Minimizes residual r_i = ||x - s_i|| - ||x - s_0|| - c*(t_i - t_0)
        """
        if init_guess is None:
            # Centroid of stations as initial guess
            x = np.mean(self.stations, axis=0) + np.array([0.0, 0.0, 100.0])
        else:
            x = np.array(init_guess, dtype=float)
            
        ref_s = self.stations[0]
        ref_t = time_of_arrivals[0]
        
        # Iterative Gauss-Newton
        for _ in range(30):
            d0 = np.linalg.norm(x - ref_s)
            if d0 < 1e-4:
                d0 = 1e-4
                
            residuals = []
            J = []
            
            for i in range(1, self.num_stations):
                di = np.linalg.norm(x - self.stations[i])
                if di < 1e-4:
                    di = 1e-4
                    
                measured_range_diff = (time_of_arrivals[i] - ref_t) * self.c
                pred_range_diff = di - d0
                residuals.append(pred_range_diff - measured_range_diff)
                
                # Jacobian row: (x - s_i)/di - (x - s_0)/d0
                grad_i = (x - self.stations[i]) / di - (x - ref_s) / d0
                J.append(grad_i)
                
            residuals = np.array(residuals)
            J = np.array(J)
            
            # Solve J * delta = -residuals
            delta, _, _, _ = np.linalg.lstsq(J, -residuals, rcond=None)
            x += delta
            
            if np.linalg.norm(delta) < 1e-4:
                break
                
        return x

class UnjammableKineticInterceptor:
    """
    Low-Cost Kinetic Drone Interceptor using True Proportional Navigation (TPN)
    and Chi-Square Outlier-Gated Error-State EKF for GPS-Denied Environments.
    """
    def __init__(self, init_pos: np.ndarray, init_vel: np.ndarray, dt: float = 0.01):
        self.pos = np.array(init_pos, dtype=float)
        self.vel = np.array(init_vel, dtype=float)
        self.dt = dt
        self.nav_constant = 4.0
        
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.05
        self.R = np.eye(3) * 0.5

    def update_es_ekf_optical_flow(self, measured_pos: np.ndarray) -> np.ndarray:
        residual = measured_pos - self.pos
        S = self.P[:3, :3] + self.R
        mahalanobis_dist = float(residual.T @ np.linalg.inv(S) @ residual)
        
        if mahalanobis_dist < 11.34:
            K = self.P[:3, :3] @ np.linalg.inv(S)
            self.pos += K @ residual
            self.P[:3, :3] = (np.eye(3) - K) @ self.P[:3, :3]
            
        return self.pos

    def compute_proportional_navigation_accel(self, target_pos: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        rel_pos = target_pos - self.pos
        rel_vel = target_vel - self.vel
        range_dist = np.linalg.norm(rel_pos)
        
        if range_dist < 0.1:
            return np.zeros(3)
            
        los_unit = rel_pos / range_dist
        los_rate = np.cross(rel_pos, rel_vel) / (range_dist ** 2)
        v_closing = -np.dot(rel_vel, los_unit)
        
        a_cmd = self.nav_constant * v_closing * np.cross(los_rate, los_unit)
        
        max_g = 147.0
        accel_mag = np.linalg.norm(a_cmd)
        if accel_mag > max_g:
            a_cmd = (a_cmd / accel_mag) * max_g
            
        return a_cmd

    def step(self, target_pos: np.ndarray, target_vel: np.ndarray) -> Dict[str, any]:
        a_cmd = self.compute_proportional_navigation_accel(target_pos, target_vel)
        self.vel += a_cmd * self.dt
        self.pos += self.vel * self.dt
        
        self.P += self.Q * self.dt
        miss_distance = np.linalg.norm(target_pos - self.pos)
        interception = miss_distance <= 1.5
        
        return {
            "interceptor_pos": self.pos.copy(),
            "interceptor_vel": self.vel.copy(),
            "commanded_accel": a_cmd,
            "miss_distance": miss_distance,
            "intercepted": interception
        }

class AegisMultiMediumDefenseSystem:
    """
    AEGIS Unified Air & Maritime Counter-UAS Platform.
    Fuses acoustic array listening, multi-target allocation, and low-cost kinetic interception.
    """
    def __init__(self):
        self.acoustic_array = AcousticSensorArrayNetwork(
            stations=np.array([
                [0.0, 0.0, 0.0],
                [2000.0, 0.0, 0.0],
                [0.0, 2000.0, 0.0],
                [2000.0, 2000.0, 50.0]
            ])
        )
        self.interceptors: List[UnjammableKineticInterceptor] = []

    def deploy_interceptor(self, launch_pos: np.ndarray, initial_heading: np.ndarray) -> int:
        interceptor = UnjammableKineticInterceptor(
            init_pos=launch_pos,
            init_vel=initial_heading * 120.0
        )
        self.interceptors.append(interceptor)
        return len(self.interceptors) - 1

    def evaluate_engagement(self, target_trajectory: List[Tuple[np.ndarray, np.ndarray]], interceptor_idx: int) -> Dict[str, any]:
        interceptor = self.interceptors[interceptor_idx]
        history = []
        kill = False
        kill_time = None
        
        for step, (t_pos, t_vel) in enumerate(target_trajectory):
            measured_pos = t_pos + np.random.normal(0, 0.2, size=3)
            interceptor.update_es_ekf_optical_flow(measured_pos)
            
            res = interceptor.step(t_pos, t_vel)
            history.append(res)
            
            if res["intercepted"]:
                kill = True
                kill_time = step * interceptor.dt
                break
                
        return {
            "kinetic_kill": kill,
            "engagement_duration_sec": kill_time,
            "final_miss_distance": history[-1]["miss_distance"],
            "steps_simulated": len(history)
        }
