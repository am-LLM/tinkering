"""Course 422: Spatial Null-Steering MVDR Beamformer for Controlled Reception Pattern Antennas (CRPA)"""
import numpy as np

class CRPAMVDRBeamformer:
    def __init__(self, num_elements: int = 4, element_spacing_wavelengths: float = 0.5):
        self.num_elements = num_elements
        self.d_lambda = element_spacing_wavelengths
        # 1D linear array along X-axis
        self.element_positions = np.arange(num_elements) * element_spacing_wavelengths

    def compute_steering_vector(self, azimuth_deg: float) -> np.ndarray:
        """Computes phase delay steering vector a(theta) for plane wave incident at azimuth."""
        az_rad = np.radians(azimuth_deg)
        phase_shifts = 2.0 * np.pi * self.element_positions * np.sin(az_rad)
        return np.exp(1j * phase_shifts)

    def synthesize_mvdr_weights(
        self,
        target_satellite_az_deg: float,
        jamming_signal_az_deg: list,
        jamming_snr_db: float = 30.0
    ) -> np.ndarray:
        """
        Minimum Variance Distortionless Response (MVDR) optimal weight vector:
        w = (R_xx^-1 * a(theta_s)) / (a(theta_s)^H * R_xx^-1 * a(theta_s))
        """
        a_s = self.compute_steering_vector(target_satellite_az_deg)
        
        # Construct spatial covariance matrix R_xx
        sigma_n = 1.0 # White noise power
        R_xx = sigma_n * np.eye(self.num_elements, dtype=complex)
        
        jam_power = 10.0 ** (jamming_snr_db / 10.0)
        for j_az in jamming_signal_az_deg:
            a_j = self.compute_steering_vector(j_az)
            R_xx += jam_power * np.outer(a_j, np.conj(a_j))
            
        # Invert covariance matrix
        R_inv = np.linalg.pinv(R_xx)
        
        # Optimal weights
        numerator = np.dot(R_inv, a_s)
        denominator = np.dot(np.conj(a_s).T, numerator)
        
        weights = numerator / (denominator + 1e-12)
        return weights

    def compute_array_gain_db(self, weights: np.ndarray, azimuth_deg: float) -> float:
        """Computes spatial array power response in dB for given azimuth."""
        a = self.compute_steering_vector(azimuth_deg)
        response = np.abs(np.dot(np.conj(weights), a)) ** 2
        return float(10.0 * np.log10(max(1e-6, response)))

    def verify_properties(self) -> bool:
        w = self.synthesize_mvdr_weights(0.0, [45.0], jamming_snr_db=30.0)
        target_gain = self.compute_array_gain_db(w, 0.0)
        jam_gain = self.compute_array_gain_db(w, 45.0)
        # Spatial null should suppress jammer by at least 20 dB relative to target
        return bool((target_gain - jam_gain) > 20.0)
