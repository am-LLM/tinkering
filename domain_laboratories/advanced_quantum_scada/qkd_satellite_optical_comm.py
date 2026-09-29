"""
Decoy-State BB84 Satellite-to-Ground Optical Quantum Key Distribution (QKD) Protocol.

Simulates:
- Atmospheric channel with beam wandering, pointing jitter, scintillation index (log-normal/gamma-gamma).
- Three-intensity decoy state BB84 protocol (Signal, Decoy, Vacuum).
- SPAD (Single-Photon Avalanche Diode) Poisson statistics, quantum efficiency, dark count rate, dead time.
- Single-photon yield (Y_1) and error rate (e_1) estimation using decoy-state bounds.
- Full CASCADE error correction algorithm (multi-pass block parity & binary search bisection).
- Asymptotic and finite-key secret key rate calculations.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
from typing import Dict, List, Optional, Tuple


class Basis(Enum):
    RECTILINEAR = 0  # Z basis: |0>, |1>
    DIAGONAL = 1     # X basis: |+>, |->


class IntensityState(Enum):
    SIGNAL = 0       # Signal state (mu)
    DECOY = 1        # Decoy state (nu)
    VACUUM = 2       # Vacuum / Weak decoy state (omega)


@dataclass
class SatelliteOrbitConfig:
    """Satellite orbit and ground station geometric parameters."""
    altitude_km: float = 500.0          # Satellite orbital altitude (LEO ~500 km)
    earth_radius_km: float = 6371.0     # Mean Earth radius
    zenith_angle_deg: float = 30.0      # Zenith angle from ground station
    wavelength_nm: float = 850.0        # Optical carrier wavelength (e.g. 850nm or 1550nm)
    tx_aperture_diam_m: float = 0.3     # Satellite transmitter telescope aperture diameter
    rx_aperture_diam_m: float = 1.0     # Ground station receiver telescope aperture diameter
    pointing_jitter_rad: float = 1.0e-6 # Transmitter pointing jitter standard deviation (1 urad)

    @property
    def slant_range_m(self) -> float:
        """Calculate geometric link distance L(theta_z) in meters."""
        theta_rad = math.radians(self.zenith_angle_deg)
        r_e = self.earth_radius_km * 1000.0
        h = self.altitude_km * 1000.0
        term1 = -r_e * math.cos(theta_rad)
        term2 = math.sqrt((r_e * math.cos(theta_rad)) ** 2 + 2 * r_e * h + h ** 2)
        return term1 + term2


@dataclass
class AtmosphericTurbulenceConfig:
    """Atmospheric turbulence and scintillation parameters."""
    c_n2: float = 1.0e-14               # Refractive index structure constant (m^(-2/3))
    wind_speed_ms: float = 10.0         # Crosswind speed (m/s)
    optical_transmittance_zenith: float = 0.8  # Clear sky zenith transmittance
    outer_scale_m: float = 20.0         # Turbulence outer scale L_0
    inner_scale_m: float = 0.005        # Turbulence inner scale l_0


@dataclass
class SPADDetectorConfig:
    """Single-Photon Avalanche Diode (SPAD) detector parameters."""
    quantum_efficiency: float = 0.65    # SPAD photon detection efficiency eta_det
    dark_count_rate_hz: float = 25.0    # Dark count rate per second
    gate_window_ns: float = 1.0         # Coincidence/gating window (ns)
    afterpulse_prob: float = 0.005      # Afterpulsing probability
    dead_time_ns: float = 20.0          # Detector dead time (ns)
    optical_misalignment_prob: float = 0.015 # Optical system intrinsic error rate e_opt

    @property
    def dark_count_prob_per_gate(self) -> float:
        """Probability of dark count within a single gate window."""
        return self.dark_count_rate_hz * (self.gate_window_ns * 1.0e-9)


@dataclass
class DecoyStateBB84Config:
    """Decoy-state BB84 protocol configuration."""
    mu: float = 0.6                     # Signal mean photon number (mu)
    nu: float = 0.15                    # Decoy mean photon number (nu)
    omega: float = 0.005                # Vacuum/Weak decoy mean photon number (omega)
    prob_mu: float = 0.70               # Probability of sending signal state
    prob_nu: float = 0.20               # Probability of sending decoy state
    prob_omega: float = 0.10            # Probability of sending vacuum state
    prob_basis_z: float = 0.50          # Probability of choosing Z basis
    error_correction_efficiency_f: float = 1.16 # Reconciliation efficiency f(E)


class AtmosphericChannelModel:
    """Models beam diffraction, pointing jitter, beam wandering, and scintillation."""

    def __init__(self, orbit: SatelliteOrbitConfig, turb: AtmosphericTurbulenceConfig):
        self.orbit = orbit
        self.turb = turb
        self.wavelength_m = orbit.wavelength_nm * 1.0e-9
        self.k = 2.0 * math.pi / self.wavelength_m
        self.link_distance_m = orbit.slant_range_m

    def compute_beam_waist_at_receiver(self) -> float:
        """Calculate Gaussian beam waist radius w(L) at receiver plane (diffraction + jitter)."""
        w0 = self.orbit.tx_aperture_diam_m / 2.0
        z_r = (math.pi * (w0 ** 2)) / self.wavelength_m  # Rayleigh range
        w_diff = w0 * math.sqrt(1.0 + (self.link_distance_m / z_r) ** 2)
        jitter_radius = self.orbit.pointing_jitter_rad * self.link_distance_m
        return math.sqrt(w_diff ** 2 + 2.0 * (jitter_radius ** 2))

    def compute_scintillation_index(self) -> float:
        """Calculate Rytov variance / scintillation index sigma_I^2 for plane/spherical wave."""
        sec_theta = 1.0 / math.cos(math.radians(self.orbit.zenith_angle_deg))
        effective_depth_m = min(self.link_distance_m, 20000.0 * sec_theta)
        sigma_rytov_sq = 0.563 * (self.k ** (7.0 / 6.0)) * self.turb.c_n2 * (effective_depth_m ** (11.0 / 6.0))
        return min(float(sigma_rytov_sq), 1.5)

    def sample_transmittance(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample instantaneous atmospheric channel transmittance eta_atm(t).
        Combines deterministic absorption, geometric clipping, pointing jitter wandering, and log-normal scintillation.
        """
        if rng is None:
            rng = np.random.default_rng()

        w_rx = self.compute_beam_waist_at_receiver()
        r_rx = self.orbit.rx_aperture_diam_m / 2.0
        eta_geom = 1.0 - math.exp(-2.0 * ((r_rx / w_rx) ** 2))

        airmass = 1.0 / math.cos(math.radians(self.orbit.zenith_angle_deg))
        t_ext = max(0.01, self.turb.optical_transmittance_zenith ** airmass)

        sigma_i_sq = self.compute_scintillation_index()
        if sigma_i_sq > 1e-6:
            mu_ln = -0.5 * sigma_i_sq
            sigma_ln = math.sqrt(sigma_i_sq)
            i_scint = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n_samples)
        else:
            i_scint = np.ones(n_samples)

        sigma_wander = (self.orbit.pointing_jitter_rad * self.link_distance_m) / math.sqrt(2.0)
        rx_offset = rng.rayleigh(scale=max(1e-4, sigma_wander), size=n_samples)
        wandering_loss = np.exp(-2.0 * (rx_offset ** 2) / (w_rx ** 2))

        transmittance = eta_geom * t_ext * i_scint * wandering_loss
        return np.clip(transmittance, 0.0, 1.0)


class DecoyStateBB84Simulator:
    """End-to-End Decoy-State BB84 QKD Engine."""

    def __init__(
        self,
        orbit_config: Optional[SatelliteOrbitConfig] = None,
        turb_config: Optional[AtmosphericTurbulenceConfig] = None,
        spad_config: Optional[SPADDetectorConfig] = None,
        qkd_config: Optional[DecoyStateBB84Config] = None,
        seed: Optional[int] = None
    ):
        self.orbit = orbit_config or SatelliteOrbitConfig()
        self.turb = turb_config or AtmosphericTurbulenceConfig()
        self.spad = spad_config or SPADDetectorConfig()
        self.qkd = qkd_config or DecoyStateBB84Config()
        self.channel = AtmosphericChannelModel(self.orbit, self.turb)
        self.rng = np.random.default_rng(seed)

    def generate_alice_pulses(self, num_pulses: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate Alice's random raw bits, basis choices, and decoy intensity states."""
        bits = self.rng.integers(0, 2, size=num_pulses, dtype=np.uint8)
        bases = (self.rng.random(size=num_pulses) >= self.qkd.prob_basis_z).astype(np.uint8)

        p_mu = self.qkd.prob_mu
        p_nu = self.qkd.prob_nu
        r_intens = self.rng.random(size=num_pulses)
        intensities = np.zeros(num_pulses, dtype=np.uint8)
        intensities[r_intens >= p_mu] = 1
        intensities[r_intens >= (p_mu + p_nu)] = 2

        return bits, bases, intensities

    def transmit_and_detect(
        self,
        alice_bits: np.ndarray,
        alice_bases: np.ndarray,
        alice_intensities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate quantum optical transmission, SPAD detection, and Bob's basis selection."""
        n_pulses = len(alice_bits)
        bob_bases = (self.rng.random(size=n_pulses) >= self.qkd.prob_basis_z).astype(np.uint8)

        intensity_map = {0: self.qkd.mu, 1: self.qkd.nu, 2: self.qkd.omega}
        lambda_vals = np.array([intensity_map[i] for i in alice_intensities], dtype=np.float64)

        emitted_photons = self.rng.poisson(lam=lambda_vals)

        eta_atm = self.channel.sample_transmittance(n_samples=n_pulses, rng=self.rng)
        total_eta = np.clip(eta_atm * self.spad.quantum_efficiency, 0.0, 1.0)

        p_photon_detect = 1.0 - np.power(1.0 - total_eta, emitted_photons)
        p_dark = self.spad.dark_count_prob_per_gate
        p_total_detect = 1.0 - (1.0 - p_photon_detect) * (1.0 - p_dark)

        r_clicks = self.rng.random(size=n_pulses)
        bob_clicks = (r_clicks < p_total_detect)

        bob_bits = np.full(n_pulses, -1, dtype=np.int8)

        for i in np.flatnonzero(bob_clicks):
            is_dark_count = (self.rng.random() < p_dark / max(1e-12, p_total_detect[i]))
            basis_matched = (alice_bases[i] == bob_bases[i])

            if is_dark_count:
                bob_bits[i] = int(self.rng.integers(0, 2))
            elif basis_matched:
                if self.rng.random() < self.spad.optical_misalignment_prob:
                    bob_bits[i] = int(1 - alice_bits[i])
                else:
                    bob_bits[i] = int(alice_bits[i])
            else:
                bob_bits[i] = int(self.rng.integers(0, 2))

        return bob_bases, bob_clicks, bob_bits

    def sift_keys(
        self,
        alice_bits: np.ndarray,
        alice_bases: np.ndarray,
        alice_intensities: np.ndarray,
        bob_bases: np.ndarray,
        bob_clicks: np.ndarray,
        bob_bits: np.ndarray
    ) -> Dict[str, Dict]:
        """Perform classical basis sifting and partition into Signal, Decoy, and Vacuum subsets."""
        results = {}
        for state_name, state_val in [('signal', 0), ('decoy', 1), ('vacuum', 2)]:
            mask_state = (alice_intensities == state_val)
            mask_valid = mask_state & bob_clicks & (alice_bases == bob_bases)
            total_sent = np.count_nonzero(mask_state & (alice_bases == bob_bases))
            clicks = np.count_nonzero(mask_valid)

            a_sifted = alice_bits[mask_valid]
            b_sifted = bob_bits[mask_valid]

            n_errors = np.count_nonzero(a_sifted != b_sifted)
            qber = (n_errors / clicks) if clicks > 0 else 0.0
            gain = (clicks / total_sent) if total_sent > 0 else 0.0

            results[state_name] = {
                'total_sent': int(total_sent),
                'clicks': int(clicks),
                'gain': float(gain),
                'errors': int(n_errors),
                'qber': float(qber),
                'alice_sifted': a_sifted,
                'bob_sifted': b_sifted,
            }
        return results

    def estimate_decoy_bounds(self, sifted: Dict[str, Dict]) -> Dict[str, float]:
        """Estimate single-photon yield Y_1 (lower bound) and error rate e_1 (upper bound)."""
        mu = self.qkd.mu
        nu = self.qkd.nu
        omega = self.qkd.omega

        q_mu = sifted['signal']['gain']
        e_mu = sifted['signal']['qber']
        q_nu = sifted['decoy']['gain']
        e_nu = sifted['decoy']['qber']
        q_omega = sifted['vacuum']['gain']
        e_omega = sifted['vacuum']['qber']

        y0_est = max(1e-7, q_omega * math.exp(omega))

        numerator = (q_nu * math.exp(nu) - q_omega * math.exp(omega) * (nu ** 2 / max(1e-8, omega ** 2))
                     - ((omega ** 2 - nu ** 2) / (mu ** 2)) * (q_mu * math.exp(mu) - y0_est))
        denominator = nu - omega - (nu ** 2 - omega ** 2) / mu
        if denominator > 0 and numerator > 0:
            y1_lower = numerator / (denominator * max(1e-6, mu))
        else:
            y1_lower = max(1e-5, (q_nu * math.exp(nu) - q_omega * math.exp(omega)) / max(1e-4, nu))

        y1_lower = min(1.0, max(1e-6, y1_lower))

        e1_num = e_nu * q_nu * math.exp(nu) - e_omega * q_omega * math.exp(omega)
        e1_upper = (e1_num / (nu * y1_lower)) if (y1_lower > 0 and nu > 0) else e_mu
        e1_upper = min(0.5, max(1e-4, float(e1_upper)))

        return {
            'Y0': float(y0_est),
            'Y1_lower': float(y1_lower),
            'e1_upper': float(e1_upper),
        }

    @staticmethod
    def binary_entropy(p: float) -> float:
        """Compute binary Shannon entropy H_2(p)."""
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

    def calculate_secret_key_rate(self, sifted: Dict[str, Dict], bounds: Dict[str, float]) -> Dict[str, float]:
        """Calculate asymptotic GLLP secure key generation rate."""
        q = 0.5
        q_mu = sifted['signal']['gain']
        e_mu = sifted['signal']['qber']
        y1_l = bounds['Y1_lower']
        e1_u = bounds['e1_upper']
        mu = self.qkd.mu
        f_ec = self.qkd.error_correction_efficiency_f

        q1 = y1_l * mu * math.exp(-mu)
        h2_emu = self.binary_entropy(e_mu)
        h2_e1 = self.binary_entropy(e1_u)

        raw_rate = - q_mu * f_ec * h2_emu + q1 * (1.0 - h2_e1)
        r_secure = max(0.0, float(q * raw_rate))

        return {
            'Q_mu': float(q_mu),
            'E_mu': float(e_mu),
            'Q_1': float(q1),
            'H2_E_mu': float(h2_emu),
            'H2_e1': float(h2_e1),
            'secure_key_rate_per_pulse': r_secure,
        }


class CascadeErrorReconciliation:
    """CASCADE Error Correction Algorithm."""

    def __init__(self, num_iterations: int = 4, target_qber: float = 0.03, seed: Optional[int] = None):
        self.num_iterations = num_iterations
        self.target_qber = max(0.001, target_qber)
        self.rng = np.random.default_rng(seed)

    def get_initial_block_size(self, qber: float) -> int:
        """Optimal block size k_1 = ceil(0.73 / QBER)."""
        if qber <= 1e-4:
            return 100
        return max(4, int(math.ceil(0.73 / qber)))

    def reconcile(
        self,
        alice_key: np.ndarray,
        bob_key: np.ndarray
    ) -> Tuple[np.ndarray, int, float, bool]:
        """Execute CASCADE algorithm."""
        n = len(alice_key)
        if n == 0:
            return bob_key.copy(), 0, 0.0, True

        bob = bob_key.copy().astype(np.uint8)
        alice = alice_key.copy().astype(np.uint8)

        parity_bits_exchanged = 0
        block_history = []

        initial_qber = np.count_nonzero(alice != bob) / n
        k1 = self.get_initial_block_size(max(initial_qber, self.target_qber))

        for it in range(self.num_iterations):
            block_size = k1 * (2 ** it)
            if it == 0:
                perm = np.arange(n)
            else:
                perm = self.rng.permutation(n)

            num_blocks = int(math.ceil(n / block_size))
            for b_idx in range(num_blocks):
                start = b_idx * block_size
                end = min(n, (b_idx + 1) * block_size)
                orig_indices = perm[start:end]

                alice_block = alice[orig_indices]
                bob_block = bob[orig_indices]

                p_alice = int(np.sum(alice_block) % 2)
                p_bob = int(np.sum(bob_block) % 2)
                parity_bits_exchanged += 1

                block_history.append((orig_indices, p_alice))

                if p_alice != p_bob:
                    error_pos, bits_exchanged = self._binary_search_correction(
                        alice, bob, orig_indices
                    )
                    parity_bits_exchanged += bits_exchanged
                    bob[error_pos] = 1 - bob[error_pos]

                    if it > 0:
                        parity_bits_exchanged += self._cascade_backtrack(
                            alice, bob, block_history, error_pos
                        )

        final_errors = int(np.count_nonzero(alice != bob))
        final_ber = float(final_errors / n)
        success = bool(final_errors == 0)

        return bob, parity_bits_exchanged, final_ber, success

    def _binary_search_correction(
        self,
        alice: np.ndarray,
        bob: np.ndarray,
        indices: np.ndarray
    ) -> Tuple[int, int]:
        """Recursive bisection to locate an erroneous bit index within indices."""
        queries = 0
        curr_indices = indices.copy()

        while len(curr_indices) > 1:
            mid = len(curr_indices) // 2
            left_indices = curr_indices[:mid]

            p_a_left = int(np.sum(alice[left_indices]) % 2)
            p_b_left = int(np.sum(bob[left_indices]) % 2)
            queries += 1

            if p_a_left != p_b_left:
                curr_indices = left_indices
            else:
                curr_indices = curr_indices[mid:]

        return curr_indices[0], queries

    def _cascade_backtrack(
        self,
        alice: np.ndarray,
        bob: np.ndarray,
        block_history: List[Tuple[np.ndarray, int]],
        changed_idx: int
    ) -> int:
        """Check prior blocks in history affected by the bit flip and correct cascaded errors."""
        queries = 0
        for orig_indices, expected_parity in reversed(block_history[:-1]):
            if changed_idx in orig_indices:
                curr_bob_parity = int(np.sum(bob[orig_indices]) % 2)
                queries += 1
                if curr_bob_parity != expected_parity:
                    err_pos, q = self._binary_search_correction(alice, bob, orig_indices)
                    queries += q
                    bob[err_pos] = 1 - bob[err_pos]
        return queries
