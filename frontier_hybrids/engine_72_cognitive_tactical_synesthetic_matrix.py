"""
Engine 72: Triune Cognitive-Tactical Synesthetic Matrix
Author: Ali Malik (@am-LLM)

Fuses 5 distinct modalities across neurobiology, sensory substitution, and electronic warfare:
1. Multi-Spectral Remote Sensing (Radar CFAR / Optical contrast)
2. Retinal / Helmet Tactical HUD Rasterization
3. Biologic Intent & Autonomic Monitoring (ACC error conflict, Vagal tone HRV, dlPFC load)
4. P300 Subconscious Evoked Potentials (300ms visual oddball detection)
5. Electro-Tactile Lingual Sensory Substitution (Tongue-matrix tactile spatial mapping)

Emergent Capability: Synesthetic Battlefield Omniscience without Cognitive Tunneling.
When combat panic or cognitive saturation collapses the pilot's visual field, the engine
dynamically diverts peripheral spatial threats to the lingual electro-tactile array,
while subconscious P300 visual spikes auto-lock CFAR radar reticles before conscious recognition.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class CognitiveTacticalSynestheticEngine:
    def __init__(self, tongue_grid_dim: int = 8):
        self.tongue_grid_dim = tongue_grid_dim # 8x8 tactile lingual matrix
        self.tongue_matrix = np.zeros((tongue_grid_dim, tongue_grid_dim), dtype=float)
        
        # Pilot Biometric History for baseline drift
        self.hrv_baseline_ms = 55.0
        self.acc_conflict_threshold = 0.70

    def decode_autonomic_saturation(self, hrv_rmssd_ms: float, acc_conflict: float, beta_power: float) -> Tuple[bool, float]:
        """
        Detects cognitive tunneling & sympathetic fight-or-flight freeze.
        Returns: (is_tunneling_active, cognitive_saturation_score)
        """
        # Low HRV indicates sympathetic freeze; high ACC conflict indicates decision paralysis
        sympathetic_stress = np.clip((self.hrv_baseline_ms - hrv_rmssd_ms) / self.hrv_baseline_ms, 0.0, 1.0)
        saturation = float(0.4 * sympathetic_stress + 0.35 * acc_conflict + 0.25 * beta_power)
        is_tunneling = (saturation > 0.65) or (acc_conflict > self.acc_conflict_threshold)
        return is_tunneling, float(np.clip(saturation, 0.0, 1.0))

    def detect_p300_subconscious_spike(self, eeg_epoch: np.ndarray, stimulus_onset_sample: int = 50) -> Tuple[bool, float]:
        """
        Matched-filter detector for P300 positive deflection (250-400ms post-stimulus).
        eeg_epoch: (channels, samples) at 250 Hz
        """
        # P300 window: samples 112 to 150 (~250ms to 400ms post stimulus at 250Hz)
        p300_start = stimulus_onset_sample + int(0.25 * 250)
        p300_end = stimulus_onset_sample + int(0.40 * 250)
        
        parietal_ch = eeg_epoch[0, :] # Pz electrode
        pre_stim = parietal_ch[:stimulus_onset_sample]
        post_stim = parietal_ch[p300_start:p300_end]
        
        baseline_noise = np.std(pre_stim) + 1e-6
        peak_amp = np.max(post_stim) - np.mean(pre_stim)
        snr = float(peak_amp / baseline_noise)
        
        has_p300 = snr > 2.2
        return has_p300, snr

    def map_threats_to_lingual_matrix(self, peripheral_threats: List[Dict[str, float]]) -> np.ndarray:
        """
        Maps 360-degree azimuth and distance threats into an 8x8 lingual electro-tactile grid.
        Center (4,4) is pilot; rows/cols represent forward/aft and lateral quadrant vectors.
        """
        matrix = np.zeros((self.tongue_grid_dim, self.tongue_grid_dim), dtype=float)
        center = self.tongue_grid_dim / 2.0
        
        for threat in peripheral_threats:
            azimuth_rad = np.radians(threat.get("azimuth_deg", 0.0))
            dist_norm = np.clip(threat.get("distance_m", 1000.0) / 2000.0, 0.1, 1.0)
            threat_severity = threat.get("severity", 1.0) # 0.0 to 1.0
            
            # Map polar to grid coordinates
            r = (1.0 - dist_norm) * (center - 0.5)
            x = int(np.clip(center + r * np.sin(azimuth_rad), 0, self.tongue_grid_dim - 1))
            y = int(np.clip(center - r * np.cos(azimuth_rad), 0, self.tongue_grid_dim - 1))
            
            matrix[y, x] = max(matrix[y, x], threat_severity)
            
        self.tongue_matrix = matrix
        return matrix

    def process_synesthetic_tactical_step(
        self,
        hrv_rmssd_ms: float,
        acc_conflict: float,
        beta_power: float,
        eeg_epoch: np.ndarray,
        cfar_radar_threats: List[Dict],
        current_reticle_target: Optional[str] = None
    ) -> Dict:
        """
        Executes unified synesthetic arbitration loop.
        """
        # 1. Evaluate cognitive saturation
        tunneling_active, sat_score = self.decode_autonomic_saturation(hrv_rmssd_ms, acc_conflict, beta_power)
        
        # 2. Check for subconscious P300 oddball detection
        p300_detected, p300_snr = self.detect_p300_subconscious_spike(eeg_epoch)
        
        # 3. Target lock arbitration
        auto_locked_target = current_reticle_target
        if p300_detected and cfar_radar_threats:
            # Subconscious detection locks closest threat
            auto_locked_target = cfar_radar_threats[0].get("id", "THREAT_CFAR_01")
            
        # 4. Modality Routing: If visual tunneling active, divert to tongue-matrix
        lingual_active = tunneling_active
        lingual_stim_grid = None
        if lingual_active:
            lingual_stim_grid = self.map_threats_to_lingual_matrix(cfar_radar_threats)
            
        return {
            "cognitive_tunneling_detected": bool(tunneling_active),
            "cognitive_saturation_score": float(sat_score),
            "p300_subconscious_target_lock": bool(p300_detected),
            "p300_snr": float(p300_snr),
            "active_locked_target_id": auto_locked_target,
            "sensory_substitution_active": bool(lingual_active),
            "lingual_matrix_density": float(np.sum(self.tongue_matrix)) if lingual_active else 0.0,
            "hud_visual_clutter_reduced": bool(tunneling_active),
            "doctrine_directive": (
                "COGNITIVE TUNNELING ACTIVE: Peripheral visual telemetry diverted to lingual tactile grid"
                if tunneling_active else
                "STANDARD MULTI-SPECTRAL RETINAL HUD NOMINAL"
            )
        }
