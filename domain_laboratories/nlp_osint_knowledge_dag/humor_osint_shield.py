"""
Humor & OSINT Shield: Micro-Acoustic Prosody Feature Extractor & NLP Sarcasm/Deception Scorer.

Features:
- Digital Signal Processing (DSP) acoustic feature extractor:
  - Fundamental frequency (F0) tracking via peak autocorrelation.
  - Pitch Jitter (period-to-period frequency perturbation).
  - Spectral Shimmer (amplitude perturbation quotient).
  - Speaking Cadence & Voice-to-Pause ratio.
  - Spectral Centroid, Spectral Spread, and Spectral Flux via STFT.
- Natural Language Processing (NLP) Sarcasm & Social Engineering Deception Engine:
  - Lexical Valence Contrast & Hyperbole Detection.
  - Acoustic-Lexical incongruity modeling (e.g. sarcastic praising with monotone or exaggerated contour).
  - Social engineering attack indicators (Authority Gradient, Urgency Pressure, Evasiveness Index).
  - Multi-modal discriminative classification of genuine vs covertly deceptive/malicious communications.
"""

from dataclasses import dataclass, field
import math
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class MicroAcousticFeatures:
    """Extracted micro-acoustic prosodic biomarkers."""
    f0_mean_hz: float            # Mean fundamental frequency (Hz)
    f0_std_hz: float             # Pitch variability / contour dynamic range
    pitch_jitter_pct: float      # Period-to-period frequency perturbation (%)
    spectral_shimmer_pct: float  # Amplitude perturbation quotient (%)
    speech_to_pause_ratio: float # Ratio of active voiced frames to silent pauses
    speaking_cadence_hz: float   # Syllabic pulse / modulation cadence (Hz)
    spectral_centroid_hz: float  # Center of mass of the spectrum
    spectral_flux: float         # Rate of spectral magnitude variation across frames


@dataclass
class NLPDeceptionScores:
    """Linguistic and conversational deception/sarcasm diagnostics."""
    sarcasm_score: float         # Lexical irony / valence contrast index in [0.0, 1.0]
    urgency_pressure_score: float# Social engineering urgency framing
    authority_gradient_score: float # Impersonation / coercive hierarchy markers
    evasiveness_index: float     # Linguistic distancing and non-committal hedging
    cognitive_load_index: float  # Structural disfluency and syntactic complexity
    lexical_valence: float       # Word-level sentiment valence [-1.0, 1.0]


@dataclass
class DeceptionAssessment:
    """Unified multi-modal social engineering and deception assessment."""
    composite_deception_score: float # Combined multi-modal risk score [0.0, 1.0]
    is_deceptive: bool               # Decision threshold verdict
    is_sarcastic: bool               # Sarcasm detection verdict
    social_engineering_risk_level: str # 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'
    acoustic_features: MicroAcousticFeatures
    nlp_scores: NLPDeceptionScores
    incongruity_delta: float         # Difference between acoustic arousal and lexical claim


class MicroAcousticProsodyExtractor:
    """DSP engine for extracting micro-acoustic prosody from raw audio waveforms."""

    def __init__(self, sample_rate_hz: int = 16000, frame_len_ms: float = 25.0, frame_step_ms: float = 10.0):
        self.fs = sample_rate_hz
        self.frame_len = int(sample_rate_hz * (frame_len_ms / 1000.0))
        self.frame_step = int(sample_rate_hz * (frame_step_ms / 1000.0))

    def extract_features(self, audio: np.ndarray) -> MicroAcousticFeatures:
        """
        Extract fundamental frequency, jitter, shimmer, cadence, and spectral moments
        from 1D float/int audio array.
        """
        if len(audio) == 0:
            return MicroAcousticFeatures(0, 0, 0, 0, 0, 0, 0, 0)

        # Normalize waveform
        sig = audio.astype(np.float64)
        if np.max(np.abs(sig)) > 0:
            sig = sig / np.max(np.abs(sig))

        # Split into short-time frames
        n_frames = max(1, int(math.floor((len(sig) - self.frame_len) / self.frame_step)) + 1)
        frames = []
        frame_energies = []

        window = np.hamming(self.frame_len)
        for i in range(n_frames):
            start = i * self.frame_step
            end = start + self.frame_len
            frame = sig[start:end]
            if len(frame) == self.frame_len:
                w_frame = frame * window
                frames.append(w_frame)
                frame_energies.append(float(np.sum(w_frame ** 2)))

        if not frames:
            return MicroAcousticFeatures(0, 0, 0, 0, 0, 0, 0, 0)

        frame_energies = np.array(frame_energies)
        energy_threshold = 0.05 * np.max(frame_energies) if np.max(frame_energies) > 0 else 0.01
        voiced_mask = frame_energies > energy_threshold

        # 1. Fundamental Frequency (F0) tracking via Autocorrelation on voiced frames
        f0_list = []
        period_lengths = []
        amplitudes = []

        min_lag = int(self.fs / 400.0)  # max F0 ~ 400 Hz
        max_lag = int(self.fs / 60.0)   # min F0 ~ 60 Hz

        for i, frame in enumerate(frames):
            if not voiced_mask[i]:
                continue

            # Normalized autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(frame) - 1 :]
            if len(corr) > max_lag:
                search_region = corr[min_lag:max_lag]
                if len(search_region) > 0 and np.max(search_region) > 0:
                    peak_lag = min_lag + np.argmax(search_region)
                    f0 = self.fs / peak_lag
                    f0_list.append(f0)
                    period_lengths.append(peak_lag)
                    amplitudes.append(np.max(np.abs(frame)))

        f0_mean = float(np.mean(f0_list)) if f0_list else 120.0
        f0_std = float(np.std(f0_list)) if len(f0_list) > 1 else 0.0

        # 2. Pitch Jitter (period-to-period cycle perturbation)
        # Jitter (%) = ( (1/(N-1)) * sum |T_i - T_{i+1}| ) / ( (1/N) * sum T_i ) * 100
        if len(period_lengths) > 2:
            periods = np.array(period_lengths, dtype=np.float64)
            diffs = np.abs(np.diff(periods))
            jitter_pct = float((np.mean(diffs) / np.mean(periods)) * 100.0)
        else:
            jitter_pct = 1.0

        # 3. Spectral Shimmer (amplitude perturbation)
        # Shimmer (%) = ( (1/(N-1)) * sum |A_i - A_{i+1}| ) / ( (1/N) * sum A_i ) * 100
        if len(amplitudes) > 2:
            amps = np.array(amplitudes, dtype=np.float64)
            a_diffs = np.abs(np.diff(amps))
            shimmer_pct = float((np.mean(a_diffs) / max(1e-6, np.mean(amps))) * 100.0)
        else:
            shimmer_pct = 3.0

        # 4. Speaking Cadence & Speech-to-Pause Ratio
        n_voiced = np.count_nonzero(voiced_mask)
        n_unvoiced = len(voiced_mask) - n_voiced
        speech_to_pause = float(n_voiced / max(1, n_unvoiced))

        # Cadence: frequency of energy envelope modulation
        duration_s = max(0.01, len(sig) / self.fs)
        energy_peaks = np.count_nonzero(
            (frame_energies[1:-1] > frame_energies[:-2]) & (frame_energies[1:-1] > frame_energies[2:])
        )
        cadence_hz = float(energy_peaks / duration_s)

        # 5. Spectral Centroid & Spectral Flux via STFT
        fft_size = 512
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / self.fs)
        spectra = []
        centroids = []

        for frame in frames:
            mag = np.abs(np.fft.rfft(frame, n=fft_size))
            mag_sum = np.sum(mag)
            if mag_sum > 1e-9:
                centroid = np.sum(freqs * mag) / mag_sum
                centroids.append(centroid)
            spectra.append(mag)

        spectral_centroid = float(np.mean(centroids)) if centroids else 1500.0

        # Spectral Flux: difference in successive normalized spectra
        flux_vals = []
        for i in range(1, len(spectra)):
            s_prev = spectra[i - 1] / max(1e-6, np.sum(spectra[i - 1]))
            s_curr = spectra[i] / max(1e-6, np.sum(spectra[i]))
            flux_vals.append(np.sqrt(np.sum((s_curr - s_prev) ** 2)))

        spectral_flux = float(np.mean(flux_vals)) if flux_vals else 0.0

        return MicroAcousticFeatures(
            f0_mean_hz=f0_mean,
            f0_std_hz=f0_std,
            pitch_jitter_pct=min(50.0, jitter_pct),
            spectral_shimmer_pct=min(50.0, shimmer_pct),
            speech_to_pause_ratio=speech_to_pause,
            speaking_cadence_hz=cadence_hz,
            spectral_centroid_hz=spectral_centroid,
            spectral_flux=spectral_flux
        )

    @staticmethod
    def synthesize_mock_speech(
        duration_s: float = 1.0,
        f0_hz: float = 150.0,
        jitter_amp: float = 0.02,
        shimmer_amp: float = 0.05,
        sample_rate_hz: int = 16000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Synthesize controllable harmonic speech-like signal for testing and calibration."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0, duration_s, int(sample_rate_hz * duration_s), endpoint=False)
        sig = np.zeros_like(t)

        # Harmonics with frequency perturbation (jitter) and amplitude perturbation (shimmer)
        harmonics = [1.0, 0.5, 0.25, 0.12]
        cum_phase = 0.0
        dt = 1.0 / sample_rate_hz

        for i in range(len(t)):
            # Add jitter to instantaneous frequency
            inst_f0 = f0_hz * (1.0 + jitter_amp * rng.normal())
            cum_phase += 2.0 * math.pi * inst_f0 * dt

            # Amplitude modulation (shimmer + syllabic envelope)
            inst_amp = (1.0 + shimmer_amp * rng.normal()) * (0.5 + 0.5 * math.sin(2.0 * math.pi * 3.0 * t[i]))

            sample = 0.0
            for h_idx, h_weight in enumerate(harmonics):
                sample += h_weight * math.sin((h_idx + 1) * cum_phase)
            sig[i] = sample * inst_amp

        # Add low-amplitude Gaussian background noise
        sig += 0.01 * rng.normal(size=len(t))
        return sig


class NLPSarcasmDeceptionAnalyzer:
    """Linguistic analyzer for sarcasm, deception, and social engineering vectors."""

    URGENCY_LEXICON = {
        "immediately", "urgent", "asap", "emergency", "wire", "transfer", "freeze",
        "suspended", "breach", "critical", "deadline", "fast", "now", "expire", "action required"
    }

    AUTHORITY_LEXICON = {
        "ceo", "director", "officer", "fbi", "police", "legal", "subpoena", "court",
        "board", "ciso", "president", "compliance", "authorized", "mandatory", "override"
    }

    HEDGING_EVASIVENESS = {
        "maybe", "perhaps", "supposedly", "as far as i know", "to my knowledge",
        "allegedly", "sort of", "kind of", "not really", "basically", "technically",
        "honestly", "trust me", "believe me", "to be honest"
    }

    SARCASTIC_CONTRASTS = [
        ("great", ["disaster", "fail", "broken", "terrible", "useless", "nightmare"]),
        ("brilliant", ["idiot", "moron", "mistake", "ruined", "incompetent"]),
        ("wonderful", ["waste", "mess", "horrible", "delay", "crash"]),
        ("perfect", ["wrong", "flawed", "garbage", "trash", "unusable"]),
        ("obviously", ["clueless", "blind", "nonsense", "joke"]),
    ]

    def __init__(self):
        pass

    def score_text(self, text: str) -> NLPDeceptionScores:
        """Compute multi-factor deception and sarcasm scores from text."""
        cleaned = text.lower().strip()
        tokens = re.findall(r'\b\w+\b', cleaned)
        n_tokens = max(1, len(tokens))

        # 1. Sarcasm / Irony Contrast Score
        sarcasm_hits = 0
        for pos_anchor, neg_companions in self.SARCASTIC_CONTRASTS:
            if pos_anchor in tokens:
                for neg in neg_companions:
                    if neg in tokens:
                        sarcasm_hits += 1

        # Quotation marks or eye-roll modifiers (e.g., "expert")
        quote_sarcasm = len(re.findall(r'["\'](?:\w+)["\']', text))
        sarcasm_score = min(1.0, (sarcasm_hits * 0.45 + quote_sarcasm * 0.25))

        # 2. Urgency Pressure Score
        urgency_hits = sum(1 for w in tokens if w in self.URGENCY_LEXICON)
        urgency_score = min(1.0, urgency_hits / max(1.0, n_tokens * 0.10))

        # 3. Authority Gradient Score
        authority_hits = sum(1 for w in tokens if w in self.AUTHORITY_LEXICON)
        authority_score = min(1.0, authority_hits / max(1.0, n_tokens * 0.08))

        # 4. Evasiveness & Hedging Index
        hedging_hits = sum(1 for phrase in self.HEDGING_EVASIVENESS if phrase in cleaned)
        evasiveness_index = min(1.0, hedging_hits * 0.3)

        # 5. Cognitive Load / Disfluency
        fillers = sum(1 for w in tokens if w in {"um", "uh", "er", "ah", "like", "you know"})
        cognitive_load = min(1.0, (fillers * 0.2 + (n_tokens / 50.0) * 0.1))

        # 6. Lexical Valence
        pos_words = {"great", "good", "excellent", "love", "wonderful", "perfect", "brilliant", "thanks"}
        neg_words = {"bad", "terrible", "hate", "fail", "broken", "awful", "scam", "wrong", "disaster"}
        n_pos = sum(1 for w in tokens if w in pos_words)
        n_neg = sum(1 for w in tokens if w in neg_words)
        lexical_valence = float(np.clip((n_pos - n_neg) / max(1.0, n_pos + n_neg), -1.0, 1.0))

        return NLPDeceptionScores(
            sarcasm_score=sarcasm_score,
            urgency_pressure_score=urgency_score,
            authority_gradient_score=authority_score,
            evasiveness_index=evasiveness_index,
            cognitive_load_index=cognitive_load,
            lexical_valence=lexical_valence
        )


class HumorOSINTShield:
    """Unified Multimodal Sarcasm, Deception & Social Engineering Defense Engine."""

    def __init__(self, sample_rate_hz: int = 16000):
        self.acoustic_extractor = MicroAcousticProsodyExtractor(sample_rate_hz)
        self.nlp_analyzer = NLPSarcasmDeceptionAnalyzer()

    def evaluate_communication(
        self,
        text: str,
        audio_waveform: Optional[np.ndarray] = None
    ) -> DeceptionAssessment:
        """
        Multimodal fusion evaluating text transcript and micro-acoustic prosody.
        Detects social engineering deception, sarcasm incongruity, and cognitive strain.
        """
        nlp_scores = self.nlp_analyzer.score_text(text)

        if audio_waveform is not None and len(audio_waveform) > 0:
            acoustic = self.acoustic_extractor.extract_features(audio_waveform)
        else:
            # Neutral baseline features if audio absent
            acoustic = MicroAcousticFeatures(
                f0_mean_hz=140.0, f0_std_hz=15.0, pitch_jitter_pct=1.2,
                spectral_shimmer_pct=3.0, speech_to_pause_ratio=2.5,
                speaking_cadence_hz=3.5, spectral_centroid_hz=1400.0, spectral_flux=0.10
            )

        # Incongruity Delta: Lexical claim vs Acoustic reality
        # E.g., High positive lexical valence (+1.0) with flat pitch (low std) or excessive jitter
        acoustic_arousal = min(1.0, (acoustic.f0_std_hz / 50.0 + acoustic.pitch_jitter_pct / 10.0) / 2.0)
        incongruity_delta = abs(nlp_scores.lexical_valence - acoustic_arousal)

        # Composite Deception Formula
        # Weighted combination of NLP social engineering vectors + acoustic stress/incongruity
        nlp_deception_part = (
            nlp_scores.urgency_pressure_score * 0.35 +
            nlp_scores.authority_gradient_score * 0.25 +
            nlp_scores.evasiveness_index * 0.20 +
            nlp_scores.sarcasm_score * 0.20
        )

        acoustic_stress_part = min(1.0, (acoustic.pitch_jitter_pct / 15.0 + acoustic.spectral_shimmer_pct / 20.0) / 2.0)
        composite_score = float(np.clip(nlp_deception_part * 0.65 + acoustic_stress_part * 0.20 + incongruity_delta * 0.15, 0.0, 1.0))

        # Classification verdicts
        is_deceptive = composite_score >= 0.45
        is_sarcastic = nlp_scores.sarcasm_score >= 0.40 or (nlp_scores.lexical_valence > 0.3 and acoustic.f0_std_hz < 8.0)

        if composite_score < 0.25:
            risk_level = "LOW"
        elif composite_score < 0.50:
            risk_level = "MODERATE"
        elif composite_score < 0.75:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return DeceptionAssessment(
            composite_deception_score=composite_score,
            is_deceptive=is_deceptive,
            is_sarcastic=is_sarcastic,
            social_engineering_risk_level=risk_level,
            acoustic_features=acoustic,
            nlp_scores=nlp_scores,
            incongruity_delta=float(incongruity_delta)
        )
