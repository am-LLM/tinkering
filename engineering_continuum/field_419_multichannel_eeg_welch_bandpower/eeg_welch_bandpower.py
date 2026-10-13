"""Course 419: Multichannel EEG Welch Power Spectral Density & Band Power Extractor"""
import numpy as np

class MultichannelEEGSpectralExtractor:
    def __init__(self, sample_rate: float = 250.0, n_fft: int = 128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "beta": (12.0, 30.0),
            "gamma": (30.0, 80.0)
        }

    def compute_welch_psd(self, signal_1d: np.ndarray) -> tuple:
        """Computes Welch Power Spectral Density with Hann windowing and 50% overlap."""
        n = len(signal_1d)
        step = self.n_fft // 2
        window = np.hanning(self.n_fft)
        win_energy = np.sum(window ** 2)
        
        psd_accum = np.zeros(self.n_fft // 2 + 1)
        count = 0
        
        for start in range(0, n - self.n_fft + 1, step):
            segment = signal_1d[start:start + self.n_fft] * window
            fft_res = np.fft.rfft(segment)
            psd_accum += np.abs(fft_res) ** 2 / (self.sample_rate * win_energy)
            count += 1
            
        psd = psd_accum / max(1, count)
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        return freqs, psd

    def extract_band_powers(self, eeg_matrix: np.ndarray) -> dict:
        """eeg_matrix: (channels, samples). Returns averaged absolute and relative band powers."""
        n_channels = eeg_matrix.shape[0]
        band_totals = {k: 0.0 for k in self.bands}
        
        for ch in range(n_channels):
            freqs, psd = self.compute_welch_psd(eeg_matrix[ch])
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            
            for band, (f_low, f_high) in self.bands.items():
                mask = (freqs >= f_low) & (freqs <= f_high)
                band_pwr = np.sum(psd[mask]) * df if np.any(mask) else 0.0
                band_totals[band] += float(band_pwr)
                
        # Average across channels
        avg_bands = {k: v / n_channels for k, v in band_totals.items()}
        total_pwr = sum(avg_bands.values()) + 1e-9
        rel_bands = {k: v / total_pwr for k, v in avg_bands.items()}
        
        return {
            "absolute_powers": avg_bands,
            "relative_powers": rel_bands,
            "theta_beta_ratio": float(avg_bands["theta"] / (avg_bands["beta"] + 1e-6)),
            "engagement_index": float(avg_bands["beta"] / (avg_bands["alpha"] + avg_bands["theta"] + 1e-6))
        }

    def verify_properties(self) -> bool:
        dummy = np.random.normal(0, 1, (4, 250))
        res = self.extract_band_powers(dummy)
        return np.isclose(sum(res["relative_powers"].values()), 1.0, atol=1e-3)
