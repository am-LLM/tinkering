"""Course 244: Surface Code Stabilizer Measurement & MWPM Syndrome Decoder"""
import numpy as np

class SurfaceCodeDistance3:
    def __init__(self):
        # 3x3 data qubit grid (9 data qubits: 0..8)
        # 4 X-stabilizers and 4 Z-stabilizers
        self.num_data = 9
        self.x_stabilizers = [
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [3, 4, 6, 7],
            [4, 5, 7, 8]
        ]
        self.z_stabilizers = [
            [0, 1],
            [2, 5],
            [3, 6],
            [7, 8]
        ]

    def measure_syndrome(self, x_errors: np.ndarray, z_errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # X-errors trigger Z-stabilizers, Z-errors trigger X-stabilizers
        z_syndrome = np.array([np.sum(x_errors[stab]) % 2 for stab in self.z_stabilizers])
        x_syndrome = np.array([np.sum(z_errors[stab]) % 2 for stab in self.x_stabilizers])
        return z_syndrome, x_syndrome

    def decode_single_bit_flip(self, z_syndrome: np.ndarray) -> np.ndarray:
        correction = np.zeros(self.num_data, dtype=int)
        if np.array_equal(z_syndrome, [1, 0, 0, 0]):
            correction[0] = 1
        elif np.array_equal(z_syndrome, [0, 1, 0, 0]):
            correction[2] = 1
        elif np.array_equal(z_syndrome, [0, 0, 1, 0]):
            correction[6] = 1
        elif np.array_equal(z_syndrome, [0, 0, 0, 1]):
            correction[8] = 1
        return correction
