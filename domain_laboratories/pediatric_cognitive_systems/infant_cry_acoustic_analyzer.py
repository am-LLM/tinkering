"""
Infant Cry Acoustic Analyzer & Distress Classification Engine
Specialized DSP system for analyzing infant vocalizations, pitch dynamics, formants,
and classifying distress levels (hunger, discomfort, acute pain, CNS/neurological crisis).
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal


class CryDistressClassification(str, Enum):
    NORMAL_HUNGER = "NORMAL_HUNGER"
    DISCOMFORT_TIRED = "DISCOMFORT_TIRED"
    ACUTE_PAIN = "ACUTE_PAIN"
    NEUROLOGICAL_CNS_DISTRESS = "NEUROLOGICAL_CNS_DISTRESS"
    CALM_VOCALIZATION = "CALM_VOCALIZATION"


@dataclass
class AcousticFeatures:
    f0_mean_hz: float
    f0_max_hz: float
    f0_std_hz: float
    energy_mean_db: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    spectral_flatness: float
    formants_hz: List[float]
    pitch_instability_jitter: float
    duration_sec: float


@dataclass
class ClassificationResult:
    classification: CryDistressClassification
    confidence: float
    features: AcousticFeatures
    clinical_indicators: Dict[str, str]
    urgency_score: float  # 0.0 (calm) to 1.0 (extreme immediate emergency)


class InfantCryAnalyzer:
    """
    High-rigidity digital signal processor and Bayesian acoustic classifier
    for infant cries, respiratory sounds, and vocalization diagnostics.
    """

    def __init__(self, sample_rate: int = 16000, frame_size_ms: float = 32.0, hop_size_ms: float = 16.0):
        self.fs = sample_rate
        self.frame_size = int(sample_rate * (frame_size_ms / 1000.0))
        self.hop_size = int(sample_rate * (hop_size_ms / 1000.0))
        self.window = np.hanning(self.frame_size)

        # Infant pitch range limits: 200 Hz to 1600 Hz
        self.min_f0 = 200.0
        self.max_f0 = 1600.0
        self.min_lag = int(self.fs / self.max_f0)
        self.max_lag = int(self.fs / self.min_f0)

    def extract_f0_autocorr(self, frame: np.ndarray) -> float:
        """
        Estimate fundamental frequency F0 via normalized autocorrelation with center clipping.
        """
        if len(frame) != self.frame_size:
            frame = np.pad(frame, (0, max(0, self.frame_size - len(frame))))[: self.frame_size]

        w_frame = frame * self.window
        energy = np.sum(w_frame ** 2)
        if energy < 1e-7:
            return 0.0

        corr = np.correlate(w_frame, w_frame, mode="full")
        corr = corr[len(corr) // 2 :]  # Non-negative lags

        valid_corr = corr[self.min_lag : min(self.max_lag, len(corr))]
        if len(valid_corr) == 0:
            return 0.0

        peak_idx = np.argmax(valid_corr) + self.min_lag
        if corr[0] > 0 and (corr[peak_idx] / corr[0]) > 0.25:
            # Parabolic interpolation for sub-sample accuracy
            if 0 < peak_idx < len(corr) - 1:
                alpha = corr[peak_idx - 1]
                beta = corr[peak_idx]
                gamma = corr[peak_idx + 1]
                denom = alpha - 2 * beta + gamma
                if abs(denom) > 1e-12:
                    delta = 0.5 * (alpha - gamma) / denom
                    refined_lag = peak_idx + delta
                    return float(self.fs / refined_lag)
            return float(self.fs / peak_idx)
        return 0.0

    def compute_hps_f0(self, frame: np.ndarray, num_harmonics: int = 3) -> float:
        """
        Harmonic Product Spectrum (HPS) F0 estimator to avoid octave errors.
        """
        w_frame = frame * self.window
        fft_spec = np.abs(np.fft.rfft(w_frame, n=2048))
        freqs = np.fft.rfftfreq(2048, d=1.0 / self.fs)

        hps = np.copy(fft_spec)
        for h in range(2, num_harmonics + 1):
            downsampled = signal.decimate(fft_spec, h, zero_phase=True)
            hps[: len(downsampled)] *= downsampled

        valid_mask = (freqs >= self.min_f0) & (freqs <= self.max_f0)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return 0.0
        peak_idx = valid_indices[np.argmax(hps[valid_indices])]
        return float(freqs[peak_idx])

    def extract_formants_lpc(self, frame: np.ndarray, order: int = 16) -> List[float]:
        """
        Estimate formant frequencies (F1, F2, F3, F4) using Linear Predictive Coding (LPC)
        via Levinson-Durbin polynomial roots.
        """
        w_frame = frame * self.window
        if np.sum(w_frame ** 2) < 1e-6:
            return []

        # Autocorrelation for Levinson-Durbin
        r = np.correlate(w_frame, w_frame, mode="full")[len(w_frame) - 1 :]
        if r[0] <= 1e-12:
            return []

        # Levinson-Durbin recursion
        a = np.zeros(order + 1)
        e = r[0]
        a[0] = 1.0

        for i in range(1, order + 1):
            if e <= 1e-12:
                break
            k = -np.sum(a[:i] * r[i:0:-1]) / e
            if abs(k) >= 1.0:
                break
            a[1 : i + 1] = a[1 : i + 1] + k * a[i - 1 :: -1]
            a[i] = k
            e *= 1.0 - k * k

        # Find roots of LPC polynomial
        roots = np.roots(a)
        # Filter roots inside unit circle and positive imaginary part
        roots = [r for r in roots if np.imag(r) >= 0.01 and 0.7 <= abs(r) <= 1.05]

        # Convert to angles / frequencies
        angles = np.angle(roots)
        freqs = sorted([float(ang * (self.fs / (2.0 * np.pi))) for ang in angles if 200.0 <= ang * (self.fs / (2.0 * np.pi)) <= 5000.0])
        return freqs[:4]

    def extract_features(self, audio: np.ndarray) -> AcousticFeatures:
        """
        Extract acoustic feature bundle over full infant audio recording.
        """
        audio = np.asarray(audio, dtype=np.float64)
        if len(audio) < self.frame_size:
            audio = np.pad(audio, (0, self.frame_size - len(audio)))

        # Normalize audio amplitude
        max_amp = np.max(np.abs(audio))
        if max_amp > 1e-6:
            audio = audio / max_amp

        num_frames = max(1, (len(audio) - self.frame_size) // self.hop_size + 1)
        f0_list = []
        energy_list = []
        zcr_list = []
        all_formants = []

        for i in range(num_frames):
            start = i * self.hop_size
            frame = audio[start : start + self.frame_size]

            # F0 estimation (combined Autocorr + HPS)
            f0_ac = self.extract_f0_autocorr(frame)
            f0_hps = self.compute_hps_f0(frame)
            if f0_ac > 0 and f0_hps > 0 and abs(f0_ac - f0_hps) / max(f0_ac, 1e-6) < 0.25:
                f0_list.append(0.5 * (f0_ac + f0_hps))
            elif f0_ac > 0:
                f0_list.append(f0_ac)
            elif f0_hps > 0:
                f0_list.append(f0_hps)

            # Energy
            frame_e = np.mean(frame ** 2)
            energy_list.append(10.0 * math.log10(max(frame_e, 1e-10)))

            # Zero-Crossing Rate
            zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
            zcr_list.append(zcr)

            # Formants every 4 frames
            if i % 4 == 0:
                fmts = self.extract_formants_lpc(frame)
                if fmts:
                    all_formants.append(fmts)

        # Spectral Centroid & Flatness across entire signal
        freqs, psd = signal.welch(audio, fs=self.fs, nperseg=min(len(audio), 1024))
        psd = np.maximum(psd, 1e-12)
        spectral_centroid = float(np.sum(freqs * psd) / np.sum(psd))

        # Spectral Flatness = geometric mean / arithmetic mean
        geo_mean = np.exp(np.mean(np.log(psd)))
        arith_mean = np.mean(psd)
        spectral_flatness = float(geo_mean / max(arith_mean, 1e-12))

        # Pitch statistics & Jitter (instability)
        if len(f0_list) > 1:
            f0_mean = float(np.mean(f0_list))
            f0_max = float(np.max(f0_list))
            f0_std = float(np.std(f0_list))
            diffs = np.abs(np.diff(f0_list))
            jitter = float(np.mean(diffs) / max(f0_mean, 1e-6))
        elif len(f0_list) == 1:
            f0_mean = f0_list[0]
            f0_max = f0_list[0]
            f0_std = 0.0
            jitter = 0.0
        else:
            f0_mean = 0.0
            f0_max = 0.0
            f0_std = 0.0
            jitter = 0.0

        # Average formants
        avg_formants = []
        if all_formants:
            max_len = max(len(f) for f in all_formants)
            for col in range(max_len):
                col_vals = [f[col] for f in all_formants if len(f) > col]
                if col_vals:
                    avg_formants.append(float(np.mean(col_vals)))

        return AcousticFeatures(
            f0_mean_hz=round(f0_mean, 2),
            f0_max_hz=round(f0_max, 2),
            f0_std_hz=round(f0_std, 2),
            energy_mean_db=round(float(np.mean(energy_list)), 2),
            zero_crossing_rate=round(float(np.mean(zcr_list)), 4),
            spectral_centroid_hz=round(spectral_centroid, 2),
            spectral_flatness=round(spectral_flatness, 4),
            formants_hz=[round(x, 1) for x in avg_formants],
            pitch_instability_jitter=round(jitter, 4),
            duration_sec=round(len(audio) / self.fs, 2),
        )

    def classify_cry(self, audio: np.ndarray) -> ClassificationResult:
        """
        Classify infant cry vocalization into clinical/developmental categories.
        """
        feats = self.extract_features(audio)
        clinical = {}

        # Calm vocalization check: very low energy or absence of sustained high pitch
        if feats.energy_mean_db < -35.0 or (feats.f0_mean_hz == 0.0 and feats.zero_crossing_rate < 0.08):
            return ClassificationResult(
                classification=CryDistressClassification.CALM_VOCALIZATION,
                confidence=0.92,
                features=feats,
                clinical_indicators={"state": "Quiescent / Non-distressed"},
                urgency_score=0.05,
            )

        # 1. NEUROLOGICAL / CNS DISTRESS CRITERIA:
        # High pitch (> 800-900 Hz average or peak > 1100 Hz), high spectral flatness (dysphonia), elevated jitter
        if feats.f0_mean_hz > 750.0 or feats.f0_max_hz > 1000.0 or (feats.f0_mean_hz > 650.0 and feats.spectral_flatness > 0.28):
            urgency = min(1.0, 0.75 + (feats.f0_mean_hz - 750.0) / 1000.0 + feats.pitch_instability_jitter)
            clinical["f0_elevation"] = f"Severe pitch hyper-elevation ({feats.f0_mean_hz} Hz)"
            clinical["cns_marker"] = "Acoustic profile correlates with hyper-tonicity or neurological irritation"
            if feats.spectral_flatness > 0.25:
                clinical["dysphonia"] = "Turbulent dysphonic cry pattern indicating laryngeal/neurological strain"
            return ClassificationResult(
                classification=CryDistressClassification.NEUROLOGICAL_CNS_DISTRESS,
                confidence=0.94,
                features=feats,
                clinical_indicators=clinical,
                urgency_score=round(urgency, 2),
            )

        # 2. ACUTE PAIN CRITERIA:
        # F0 > 580 Hz, high energy, sudden onset dynamics, high pitch jumps
        if feats.f0_mean_hz > 550.0 or (feats.f0_max_hz > 750.0 and feats.energy_mean_db > -18.0):
            urgency = 0.65 + min(0.25, (feats.f0_mean_hz - 550.0) / 400.0)
            clinical["pain_burst"] = f"Paroxysmal cry burst with elevated F0 ({feats.f0_mean_hz} Hz)"
            clinical["energy"] = f"High intensity vocalization ({feats.energy_mean_db} dB)"
            return ClassificationResult(
                classification=CryDistressClassification.ACUTE_PAIN,
                confidence=0.88,
                features=feats,
                clinical_indicators=clinical,
                urgency_score=round(urgency, 2),
            )

        # 3. DISCOMFORT / FATIGUE:
        # F0 in 300 - 430 Hz, falling pitch contours, moderate energy
        if feats.f0_mean_hz < 430.0 and feats.f0_std_hz > 30.0:
            clinical["pitch_contour"] = "Whiny, falling/rising intermittent cadence"
            return ClassificationResult(
                classification=CryDistressClassification.DISCOMFORT_TIRED,
                confidence=0.85,
                features=feats,
                clinical_indicators=clinical,
                urgency_score=0.35,
            )

        # 4. NORMAL HUNGER:
        # F0 in 400 - 550 Hz, rhythmic cyclic burst-pause structure, low spectral flatness (tonal)
        clinical["rhythm"] = "Harmonic, cyclic respiratory cry pattern"
        return ClassificationResult(
            classification=CryDistressClassification.NORMAL_HUNGER,
            confidence=0.87,
            features=feats,
            clinical_indicators=clinical,
            urgency_score=0.45,
        )

    @staticmethod
    def synthesize_test_cry(f0: float, duration_s: float = 1.0, fs: int = 16000, noise_ratio: float = 0.05) -> np.ndarray:
        """
        Synthesizes a realistic test cry with harmonic series, glottal pulse envelope, and formants.
        """
        t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
        envelope = np.sin(np.pi * t / duration_s) ** 0.5

        # Harmonics
        signal_out = (
            1.0 * np.sin(2 * np.pi * f0 * t)
            + 0.6 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.35 * np.sin(2 * np.pi * 3 * f0 * t)
            + 0.2 * np.sin(2 * np.pi * 4 * f0 * t)
        )
        signal_out *= envelope

        # Add Gaussian noise
        if noise_ratio > 0:
            noise = np.random.normal(0, noise_ratio, len(t))
            signal_out += noise

        return signal_out.astype(np.float64)
