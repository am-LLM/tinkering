"""Course 423: TDoA RF Emitter Multilateration & Passive Radar Direction Finder"""
import numpy as np

class TDoARFEmitterGeolocator:
    def __init__(self, speed_of_light_m_s: float = 299792458.0):
        self.c = speed_of_light_m_s
        # 3 mobile reconnaissance airborne nodes
        self.receiver_nodes = np.array([
            [0.0, 0.0, 100.0],
            [2000.0, 0.0, 120.0],
            [1000.0, 1732.0, 110.0]
        ])

    def compute_emitter_2d_fix(self, tdoa_timestamps_seconds: list) -> np.ndarray:
        """
        Hyperbolic multilateration solving for ground RF transmitter (x, y) location.
        tdoa_timestamps_seconds: [t_node1 - t_node0, t_node2 - t_node0]
        """
        tau_10 = tdoa_timestamps_seconds[0]
        tau_20 = tdoa_timestamps_seconds[1]
        
        range_diff_10 = tau_10 * self.c
        range_diff_20 = tau_20 * self.c
        
        s0 = self.receiver_nodes[0][:2]
        s1 = self.receiver_nodes[1][:2]
        s2 = self.receiver_nodes[2][:2]
        
        # Grid search over 4km x 4km sector for minimum squared range error
        best_pos = np.array([1000.0, 800.0])
        min_error = float("inf")
        
        x_grid = np.linspace(-1000.0, 3000.0, 50)
        y_grid = np.linspace(-1000.0, 3000.0, 50)
        
        for x in x_grid:
            for y in y_grid:
                pt = np.array([x, y])
                d0 = np.linalg.norm(pt - s0)
                d1 = np.linalg.norm(pt - s1)
                d2 = np.linalg.norm(pt - s2)
                
                err1 = (d1 - d0) - range_diff_10
                err2 = (d2 - d0) - range_diff_20
                total_err = err1 ** 2 + err2 ** 2
                
                if total_err < min_error:
                    min_error = total_err
                    best_pos = pt
                    
        return best_pos

    def verify_properties(self) -> bool:
        test_emitter = np.array([1200.0, 600.0])
        s0 = self.receiver_nodes[0][:2]
        s1 = self.receiver_nodes[1][:2]
        s2 = self.receiver_nodes[2][:2]
        
        tau_10 = (np.linalg.norm(test_emitter - s1) - np.linalg.norm(test_emitter - s0)) / self.c
        tau_20 = (np.linalg.norm(test_emitter - s2) - np.linalg.norm(test_emitter - s0)) / self.c
        
        fix = self.compute_emitter_2d_fix([tau_10, tau_20])
        return bool(np.linalg.norm(fix - test_emitter) < 100.0)
