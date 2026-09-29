"""Course 287: Surface EMG Feature Extraction & Prosthetic Gesture Classifier"""
import numpy as np

class EMGProstheticClassifier:
    def __init__(self, sample_rate: int = 1000, window_size: int = 200):
        self.fs = sample_rate
        self.win = window_size
        self.centroids = None

    def extract_mav_rms_wl(self, emg_signal: np.ndarray) -> np.ndarray:
        mav = np.mean(np.abs(emg_signal))
        rms = np.sqrt(np.mean(emg_signal**2))
        wl = np.sum(np.abs(np.diff(emg_signal)))
        return np.array([mav, rms, wl])

    def train_lda(self, training_data: dict[str, list[np.ndarray]]):
        self.centroids = {}
        for gesture, samples in training_data.items():
            feats = [self.extract_mav_rms_wl(s) for s in samples]
            self.centroids[gesture] = np.mean(feats, axis=0)

    def predict_gesture(self, emg_signal: np.ndarray) -> str:
        feats = self.extract_mav_rms_wl(emg_signal)
        best_gesture = None
        min_dist = float('inf')
        for gesture, center in self.centroids.items():
            d = np.linalg.norm(feats - center)
            if d < min_dist:
                min_dist = d
                best_gesture = gesture
        return best_gesture
