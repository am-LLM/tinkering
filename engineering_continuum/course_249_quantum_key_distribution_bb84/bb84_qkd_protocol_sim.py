"""Course 249: BB84 Quantum Key Distribution Simulation with QBER Analysis"""
import numpy as np

class BB84QKDProtocol:
    def __init__(self, n_bits: int = 1000, eve_present: bool = False, channel_noise_error_rate: float = 0.02):
        self.n = n_bits
        self.eve = eve_present
        self.noise = channel_noise_error_rate

    def run_protocol(self) -> dict:
        alice_bits = np.random.randint(0, 2, self.n)
        alice_bases = np.random.randint(0, 2, self.n) # 0: Rectilinear (+), 1: Diagonal (x)
        
        # State transmission
        bob_bases = np.random.randint(0, 2, self.n)
        bob_bits = np.zeros(self.n, dtype=int)
        
        for i in range(self.n):
            bit = alice_bits[i]
            base = alice_bases[i]
            if self.eve:
                eve_base = np.random.randint(0, 2)
                if eve_base != base:
                    bit = np.random.randint(0, 2)
            if bob_bases[i] == base:
                measured = bit if np.random.rand() > self.noise else 1 - bit
            else:
                measured = np.random.randint(0, 2)
            bob_bits[i] = measured
            
        # Sifting
        sift_mask = (alice_bases == bob_bases)
        alice_sifted = alice_bits[sift_mask]
        bob_sifted = bob_bits[sift_mask]
        
        errors = np.sum(alice_sifted != bob_sifted)
        qber = errors / float(len(alice_sifted)) if len(alice_sifted) > 0 else 0.0
        return {
            "sifted_length": len(alice_sifted),
            "errors": int(errors),
            "qber": float(qber),
            "secure": bool(qber < 0.11) # Theoretical Shor-Preskill threshold
        }
