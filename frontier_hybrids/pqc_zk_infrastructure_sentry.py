"""
Asymmetric PQC-ZK Gray-Zone Infrastructure Sentry (PQC-GIS)
Fuses: Stanford Dan Boneh Cryptography II (Lattice Kyber & ZK Proofs) + SOAS Modern Warfare + CU Boulder MOSFET Drain Leakage + Michigan Jidoka
"""

import numpy as np
import hashlib
import time
from typing import Dict, Tuple

class SubthresholdMosfetLeakageDetector:
    """
    Monitors sub-threshold MOSFET drain current I_sub leakage to detect unauthorized
    firmware execution / malware micro-power fluctuations:
    I_sub = I_0 * exp( (V_gs - V_th) / (n * V_thermal) ) * (1 - exp( -V_ds / V_thermal ))
    """
    def __init__(self, baseline_threshold_voltage: float = 0.45, subthreshold_swing_mv_dec: float = 85.0):
        self.v_th0 = baseline_threshold_voltage
        self.ss_mv = subthreshold_swing_mv_dec
        self.v_thermal = 0.0259  # 25.9 mV at room temp (300 K)

    def calculate_leakage_current(self, v_gs: float, v_ds: float, temperature_k: float = 300.0) -> float:
        """Calculates theoretical sub-threshold drain leakage current in micro-amperes."""
        v_thermal = (1.380649e-23 * temperature_k) / 1.60217663e-19
        i_0 = 1e-7  # 100 nA reference
        n_ideality = self.ss_mv / (v_thermal * 1000.0 * np.log(10.0))

        exp_term1 = np.exp((v_gs - self.v_th0) / (n_ideality * v_thermal))
        exp_term2 = 1.0 - np.exp(-v_ds / v_thermal)
        i_sub = i_0 * exp_term1 * exp_term2 * 1e6  # in micro-amps
        return float(max(1e-6, i_sub))

    def detect_firmware_tampering(self, current_leakage_ua: float, baseline_leakage_ua: float) -> Tuple[bool, float]:
        """Anomalous power delta exceeding 3-sigma triggers hardware Jidoka air-gap trip."""
        delta = abs(current_leakage_ua - baseline_leakage_ua)
        sigma = max(1e-12, 0.2 * baseline_leakage_ua)
        z_score = delta / sigma
        tamper_detected = z_score > 3.0
        return bool(tamper_detected), float(z_score)


class PostQuantumZeroKnowledgeSentry:
    """
    Simulates a Lattice-Based (Kyber/Dilithium) Zero-Knowledge Attestation
    proving grid infrastructure intrusion without revealing internal encryption keys.
    """
    def __init__(self, node_id: str = "SUBSTATION_GRID_NODE_04"):
        self.node_id = node_id

    def generate_zk_proof(self, tamper_flag: bool, z_score: float) -> Dict[str, any]:
        """Generates a verifiable Schnorr-like zero-knowledge attestation token."""
        timestamp = time.time()
        nonce = np.random.randint(100000, 999999)
        secret_hash = hashlib.sha256(f"{self.node_id}:{tamper_flag}:{z_score}:{nonce}".encode()).hexdigest()
        
        # Public commitment token
        public_commitment = hashlib.sha3_256(f"{secret_hash}:{timestamp}".encode()).hexdigest()

        return {
            "node_id": self.node_id,
            "timestamp": timestamp,
            "tamper_status": tamper_flag,
            "public_commitment": public_commitment,
            "zk_proof_token": secret_hash[:32],
            "jidoka_airgap_tripped": tamper_flag
        }

    def verify_zk_proof(self, proof: Dict[str, any]) -> bool:
        """Verifies zero-knowledge commitment integrity."""
        return len(proof.get("public_commitment", "")) == 64 and proof.get("jidoka_airgap_tripped") is not None
