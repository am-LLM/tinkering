"""
Trauma Damage Control Triage & Hemodynamic Crisis Simulator.

Implements:
1. Dynamic physiological cardiovascular engine (HR, SBP, DBP, MAP, Cardiac Output, DO2).
2. The Trauma Lethal Triad (Hypothermia, Acidosis, Trauma-Induced Coagulopathy [TIC]).
3. Damage Control Resuscitation (DCR): 1:1:1 Balanced Transfusion (PRBC:FFP:PLT) vs Crystalloid Hemodilution.
4. Hemostatic interventions (TXA, Surgical packing/tourniquet).
5. Triage classification (START / SALT / NATO Triage Tags) & Survival Probability estimation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class TriageCategory(Enum):
    GREEN = "Minimal / Delayed (Green)"
    YELLOW = "Urgent (Yellow)"
    RED = "Immediate Life Threat (Red)"
    BLACK = "Expectant / Deceased (Black)"


class ResuscitationFluidType(Enum):
    BALANCED_1_1_1 = "balanced_blood_1_1_1"  # PRBC : FFP : Platelets (1:1:1)
    PRBC_ONLY = "prbc_only"
    CRYSTALLOID_SALINE = "crystalloid_saline" # Normal saline / Hartmann's
    WHOLE_BLOOD = "warm_whole_blood"


@dataclass
class ResuscitationEvent:
    """Discrete intervention during resuscitation timeline."""
    time_min: float
    fluid_type: ResuscitationFluidType
    volume_ml: float
    fluid_temperature_c: float = 37.0  # warmed vs cold (20 C)
    give_txa: bool = False             # 1g Tranexamic acid bolus
    apply_surgical_hemostasis: bool = False # Percent reduction in bleeding rate (e.g. 90%)
    hemostasis_efficacy: float = 0.90


@dataclass
class PatientBaseline:
    """Baseline physiological parameters for a 70 kg patient."""
    weight_kg: float = 70.0
    total_blood_volume_ml: float = 5000.0  # ~70 mL/kg
    baseline_hr: float = 72.0              # bpm
    baseline_sbp: float = 120.0            # mmHg
    baseline_dbp: float = 80.0             # mmHg
    baseline_temp_c: float = 37.0          # °C
    baseline_ph: float = 7.40
    baseline_hemoglobin: float = 14.0      # g/dL
    baseline_fibrinogen: float = 300.0     # mg/dL
    baseline_platelets: float = 250.0      # x10^3 / uL
    baseline_inr: float = 1.0


@dataclass
class PhysiologicalSnapshot:
    """Physiological state at a specific simulation timestep."""
    time_min: float
    blood_volume_ml: float
    heart_rate_bpm: float
    sbp_mmhg: float
    dbp_mmhg: float
    map_mmhg: float
    shock_index: float
    temperature_c: float
    arterial_ph: float
    lactate_mmol_l: float
    base_deficit_meq_l: float
    hemoglobin_g_dl: float
    fibrinogen_mg_dl: float
    platelets_k_ul: float
    inr: float
    active_bleed_rate_ml_min: float
    total_blood_lost_ml: float
    do2_oxygen_delivery_ml_min: float
    lethal_triad_severity_index: float
    triage_tag: TriageCategory
    survival_probability: float


class TraumaHemodynamicSimulator:
    """
    Continuous physiological crisis simulator modeling acute hemorrhagic shock,
    coagulopathy escalation, and resuscitation response curves.
    """

    def __init__(self, baseline: Optional[PatientBaseline] = None):
        self.baseline = baseline or PatientBaseline()

    def calculate_triad_severity(
        self, temp_c: float, ph: float, inr: float, fibrinogen: float
    ) -> float:
        """
        Calculates Lethal Triad Severity Index (0.0 normal -> 1.0 fatal).
        Hypothermia: Temp < 35 C
        Acidosis: pH < 7.2
        Coagulopathy: INR > 1.5 or Fibrinogen < 150 mg/dL
        """
        # Hypothermia penalty
        t_penalty = max(0.0, (36.5 - temp_c) / 4.5)  # 32 C -> 1.0
        # Acidosis penalty
        ph_penalty = max(0.0, (7.35 - ph) / 0.45)    # 6.9 -> 1.0
        # Coagulopathy penalty
        inr_penalty = max(0.0, (inr - 1.2) / 1.8)     # 3.0 -> 1.0
        fib_penalty = max(0.0, (200.0 - fibrinogen) / 150.0) # 50 mg/dL -> 1.0
        c_penalty = max(inr_penalty, fib_penalty)

        triad_index = float(np.clip((0.30 * t_penalty + 0.35 * ph_penalty + 0.35 * c_penalty), 0.0, 1.0))
        return triad_index

    def determine_triage_category(
        self, hr: float, sbp: float, ph: float, bv_ratio: float, triad_idx: float
    ) -> Tuple[TriageCategory, float]:
        """
        Assigns triage tag (Green, Yellow, Red, Black) and survival probability.
        """
        # Survival probability logistic curve
        z = (
            3.5
            - 4.0 * (1.0 - bv_ratio)
            - 3.5 * triad_idx
            - 0.02 * max(0.0, hr - 100.0)
            - 0.03 * max(0.0, 90.0 - sbp)
        )
        p_surv = float(1.0 / (1.0 + np.exp(-z)))

        if p_surv < 0.05 or sbp < 35.0 or ph < 6.85:
            tag = TriageCategory.BLACK
        elif sbp < 90.0 or hr > 120.0 or bv_ratio < 0.75 or triad_idx > 0.40:
            tag = TriageCategory.RED
        elif bv_ratio < 0.90 or hr > 100.0:
            tag = TriageCategory.YELLOW
        else:
            tag = TriageCategory.GREEN

        return tag, p_surv

    def simulate(
        self,
        duration_min: float = 60.0,
        dt_min: float = 0.1,
        initial_bleed_rate_ml_min: float = 120.0,
        events: Optional[List[ResuscitationEvent]] = None,
    ) -> List[PhysiologicalSnapshot]:
        """
        Runs numerical forward integration of hemodynamic and metabolic state.
        """
        if duration_min <= 0 or dt_min <= 0:
            raise ValueError("Duration and dt must be strictly positive.")

        events = events or []
        events_sorted = sorted(events, key=lambda e: e.time_min)

        steps = int(np.ceil(duration_min / dt_min)) + 1
        timeline: List[PhysiologicalSnapshot] = []

        # Current patient state
        bv = self.baseline.total_blood_volume_ml
        hb = self.baseline.baseline_hemoglobin
        fib = self.baseline.baseline_fibrinogen
        plt = self.baseline.baseline_platelets
        inr = self.baseline.baseline_inr
        temp_c = self.baseline.baseline_temp_c
        ph = self.baseline.baseline_ph
        lactate = 1.0  # mmol/L normal
        base_def = 0.0 # meq/L normal

        base_bleed_intensity = initial_bleed_rate_ml_min
        total_blood_lost = 0.0
        txa_active = False

        event_idx = 0

        for step in range(steps):
            t = step * dt_min

            # Process discrete events occurring at or before this time
            while event_idx < len(events_sorted) and events_sorted[event_idx].time_min <= t:
                ev = events_sorted[event_idx]
                if ev.give_txa:
                    txa_active = True
                if ev.apply_surgical_hemostasis:
                    base_bleed_intensity *= (1.0 - ev.hemostasis_efficacy)

                # Fluid infusion effect
                v_inf = ev.volume_ml
                if ev.fluid_type in (ResuscitationFluidType.BALANCED_1_1_1, ResuscitationFluidType.WHOLE_BLOOD):
                    new_bv = bv + v_inf
                    hb = (hb * bv + 11.5 * v_inf) / new_bv
                    fib = (fib * bv + 260.0 * v_inf) / new_bv
                    plt = (plt * bv + 220.0 * v_inf) / new_bv
                    inr = max(1.0, (inr * bv + 1.1 * v_inf) / new_bv)
                    bv = new_bv
                elif ev.fluid_type == ResuscitationFluidType.PRBC_ONLY:
                    new_bv = bv + v_inf
                    hb = (hb * bv + 20.0 * v_inf) / new_bv
                    fib = (fib * bv) / new_bv
                    plt = (plt * bv) / new_bv
                    inr += 0.25 * (v_inf / 1000.0)
                    bv = new_bv
                elif ev.fluid_type == ResuscitationFluidType.CRYSTALLOID_SALINE:
                    expansion = v_inf * 0.30
                    new_bv = bv + expansion
                    hb = (hb * bv) / new_bv
                    fib = (fib * bv) / new_bv
                    plt = (plt * bv) / new_bv
                    inr += 0.35 * (v_inf / 1000.0)
                    ph -= 0.04 * (v_inf / 1000.0)
                    bv = new_bv

                # Temperature effect of un-warmed fluid
                if ev.fluid_temperature_c < 36.0:
                    delta_t = ((ev.fluid_temperature_c - temp_c) * (v_inf / 1000.0) * 0.8) / (self.baseline.weight_kg * 0.83)
                    temp_c += delta_t

                event_idx += 1

            # Hemodynamics calculation
            bv_ratio = np.clip(bv / self.baseline.total_blood_volume_ml, 0.1, 1.3)

            # Baroreceptor heart rate reflex
            hr_compensation = max(0.0, (1.0 - bv_ratio) * 110.0)
            hr = float(np.clip(self.baseline.baseline_hr + hr_compensation, 30.0, 180.0))

            # Blood Pressure (Systolic, Diastolic, MAP)
            sbp = float(np.clip(self.baseline.baseline_sbp * (bv_ratio ** 1.3), 20.0, 170.0))
            dbp = float(np.clip(self.baseline.baseline_dbp * (bv_ratio ** 1.1), 10.0, 110.0))
            map_val = (sbp + 2.0 * dbp) / 3.0
            shock_index = float(hr / max(1.0, sbp))

            # Cardiac Output (L/min) ~ Stroke Volume * HR
            stroke_vol_ml = max(10.0, 70.0 * bv_ratio)
            cardiac_output_l_min = (stroke_vol_ml * hr) / 1000.0

            # Oxygen delivery DO2 (mL O2 / min) = CO (dL/min) * 1.34 * Hb * SaO2
            sao2 = 0.98 if sbp > 60 else max(0.70, 0.98 - (60.0 - sbp) * 0.005)
            do2 = cardiac_output_l_min * 10.0 * 1.34 * max(1.0, hb) * sao2

            # Coagulopathy & Bleed Rate
            coag_factor_temp = np.exp(0.08 * (temp_c - 37.0))
            coag_factor_ph = 1.0 if ph >= 7.35 else max(0.2, 1.0 - 2.5 * (7.35 - ph))
            clotting_efficiency = coag_factor_temp * coag_factor_ph * min(1.5, fib / 200.0) / max(0.8, inr)
            if txa_active:
                clotting_efficiency *= 1.35

            # Dynamic Bleed Rate
            if base_bleed_intensity > 0.1:
                pressure_factor = map_val / 80.0
                active_bleed = float(max(0.0, base_bleed_intensity * pressure_factor * (1.0 / max(0.3, clotting_efficiency))))
            else:
                active_bleed = 0.0

            # State integration for time step dt
            loss_this_step = active_bleed * dt_min
            bv = max(500.0, bv - loss_this_step)
            total_blood_lost += loss_this_step

            # Hypoperfusion -> Anaerobic metabolism -> Lactate & Acidosis
            hypoperfusion = max(0.0, (65.0 - map_val) / 65.0) + max(0.0, (0.80 - bv_ratio) / 0.80)
            if hypoperfusion > 0.05 or do2 < 600.0:
                lactate_prod_rate = max(0.1, hypoperfusion * 0.45)
                lactate += lactate_prod_rate * dt_min
                base_def += lactate_prod_rate * dt_min * 1.2
                ph = max(6.80, self.baseline.baseline_ph - 0.035 * base_def)
            else:
                # Clearance when resuscitated
                lactate = max(1.0, lactate - 0.05 * dt_min)
                base_def = max(0.0, base_def - 0.04 * dt_min)
                ph = min(self.baseline.baseline_ph, ph + 0.005 * dt_min)

            # Passive body cooling under shock
            ambient_temp_c = 22.0
            cooling_rate = 0.015 * (1.0 - min(1.0, do2 / 800.0)) * (temp_c - ambient_temp_c)
            temp_c = max(30.0, temp_c - cooling_rate * dt_min)

            # Triad and Triage assessment
            triad_idx = self.calculate_triad_severity(temp_c, ph, inr, fib)
            tag, p_surv = self.determine_triage_category(hr, sbp, ph, bv_ratio, triad_idx)

            snapshot = PhysiologicalSnapshot(
                time_min=round(t, 2),
                blood_volume_ml=float(bv),
                heart_rate_bpm=hr,
                sbp_mmhg=sbp,
                dbp_mmhg=dbp,
                map_mmhg=map_val,
                shock_index=shock_index,
                temperature_c=float(temp_c),
                arterial_ph=float(ph),
                lactate_mmol_l=float(lactate),
                base_deficit_meq_l=float(base_def),
                hemoglobin_g_dl=float(hb),
                fibrinogen_mg_dl=float(fib),
                platelets_k_ul=float(plt),
                inr=float(inr),
                active_bleed_rate_ml_min=active_bleed,
                total_blood_lost_ml=float(total_blood_lost),
                do2_oxygen_delivery_ml_min=float(do2),
                lethal_triad_severity_index=triad_idx,
                triage_tag=tag,
                survival_probability=p_surv,
            )
            timeline.append(snapshot)

        return timeline
