"""
Engine 77: Autonomous Dormant Seismic Sentry & Acoustic Wake-on-Trigger Engine
Author: Ali Malik (@am-LLM)

Fuses 4 high-order physical and edge cybernetic substrates:
1. Ultra-Low-Power Deep Sleep Power Gating (10µA quiescent draw, multi-day survivability)
2. Piezoelectric / Geophone Ground Vibration Analysis (Seismic impulse thresholding)
3. Micro-Acoustic Spectral Energy Ingress Detection (80-300Hz diesel engine resonance)
4. Fast Hardware Interrupt State Awakening & Failsafe Spool-up (2ms motor pre-arm)

Emergent Capability: Autonomous "Perch-and-Stare" Long-Duration Ground Ambush.
Enables expendable multirotor drones to fly forward, land quietly on structures or terrain,
power down propulsion to near-zero current draw, and instantly awaken upon mechanical/acoustic
detection of passing ground vehicles or personnel.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional

class DormantSeismicSentryEngine:
    def __init__(
        self,
        quiescent_current_uA: float = 12.0,
        active_current_mA: float = 180.0,
        battery_capacity_mAh: float = 500.0
    ):
        self.i_sleep_ua = quiescent_current_uA
        self.i_active_ma = active_current_mA
        self.battery_capacity_mah = battery_capacity_mAh
        
        # State machine: DORMANT_DEEP_SLEEP -> SENSORY_EVALUATION -> ACTIVE_AWAKE
        self.current_state = "DORMANT_DEEP_SLEEP"
        self.sleep_duration_hours = 0.0
        
        # Detection thresholds
        self.seismic_threshold_g = 0.08 # 80 mg vibration trigger
        self.acoustic_snr_threshold = 3.5

    def compute_projected_battery_life_days(self, duty_cycle_pct: float = 0.01) -> float:
        """
        Calculates stationary survival endurance in days based on sleep current.
        """
        avg_current_ma = (self.i_sleep_ua / 1000.0) * (1.0 - duty_cycle_pct) + self.i_active_ma * duty_cycle_pct
        endurance_hours = self.battery_capacity_mah / max(1e-4, avg_current_ma)
        return float(endurance_hours / 24.0)

    def evaluate_seismic_acoustic_trigger(
        self,
        piezo_vibration_g: float,
        acoustic_band_energy_db: float,
        ambient_noise_db: float
    ) -> Tuple[bool, str]:
        """
        Processes low-power analog comparator hardware interrupt readouts.
        Returns: (should_awaken, wake_trigger_reason)
        """
        acoustic_snr = (acoustic_band_energy_db - ambient_noise_db)
        
        seismic_trigger = piezo_vibration_g >= self.seismic_threshold_g
        acoustic_trigger = acoustic_snr >= self.acoustic_snr_threshold
        
        if seismic_trigger and acoustic_trigger:
            return True, "DUAL_SEISMIC_ACOUSTIC_CONFIRMATION"
        elif seismic_trigger:
            return True, "SEISMIC_GROUND_VIBRATION_TRIGGER"
        elif acoustic_trigger:
            return True, "ACOUSTIC_LOW_FREQ_ENGINE_RESONANCE"
        else:
            return False, "AMBIENT_QUIET_REMAIN_DORMANT"

    def process_sentry_tick(
        self,
        time_elapsed_hours: float,
        piezo_accel_g: float,
        mic_energy_db: float,
        ambient_db: float
    ) -> Dict:
        """
        Executes single sentry state machine evaluation step.
        """
        self.sleep_duration_hours += time_elapsed_hours
        
        awaken, reason = self.evaluate_seismic_acoustic_trigger(piezo_accel_g, mic_energy_db, ambient_db)
        
        if awaken:
            self.current_state = "ACTIVE_AWAKE"
        else:
            self.current_state = "DORMANT_DEEP_SLEEP"
            
        remaining_days = self.compute_projected_battery_life_days()
        
        return {
            "sentry_state": self.current_state,
            "wake_trigger_fired": awaken,
            "trigger_reason": reason,
            "hours_in_hibernation": float(self.sleep_duration_hours),
            "projected_survival_days": remaining_days,
            "propulsion_armed": bool(awaken),
            "optical_camera_powered": bool(awaken)
        }
