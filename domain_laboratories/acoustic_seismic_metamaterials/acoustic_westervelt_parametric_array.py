r"""
Nonlinear Acoustic Westervelt Equation & Parametric Array Solver.

Simulates the generation of directional difference-frequency audio beams (audio spotlighting)
from intense dual-carrier ultrasonic transducers via nonlinear self-demodulation in air.

Theoretical Foundations:
-----------------------
1. Westervelt Equation:
   
abla^2 p - (1/c0^2) \partial^2 p/\partial t^2 + (\delta/c0^4) \partial^3 p/\partial t^3
   + (eta / (
ho0 * c0^4)) \partial^2 (p^2)/\partial t^2 = 0

2. Berktay's Far-field Self-Demodulation:
   p_audio(t, z) = (beta * S / (16 * pi * rho0 * c0^4 * alpha_u * z)) * d^2/dt^2 [ E^2(t - z/c0) ]
   where E(t) is the ultrasonic envelope modulation.

3. Numerical Methods:
   - 1D / Axisymmetric progressive Westervelt / Burgers split-step & FDTD solvers
   - Nonlinear harmonic generation (2*f1, 2*f2, f1+f2) & parametric downconversion (fd = |f2 - f1|)
   - Spatial directivity / beam pattern evaluation (pencil beam characteristics)
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import Dict, Tuple, Optional, List


@dataclass
class AirMediumProperties:
    """Acoustic and thermodynamic properties of ambient air."""
    c0: float = 343.2          # Ambient speed of sound (m/s) at 20°C
    rho0: float = 1.204        # Equilibrium air density (kg/m^3) at 1 atm, 20°C
    gamma: float = 1.40        # Ratio of specific heats for dry air
    beta: float = 1.20         # Parameter of nonlinearity beta = 1 + B/(2A) = (gamma + 1)/2 = 1.20
    delta: float = 3.9e-5      # Acoustic diffusivity (m^2/s) associated with thermoviscous absorption
    ambient_pressure: float = 101325.0  # P0 in Pa

    @property
    def acoustic_impedance(self) -> float:
        """Specific acoustic impedance z0 = rho0 * c0 (Pa·s/m)."""
        return self.rho0 * self.c0

    def absorption_coefficient(self, freq_hz: float) -> float:
        """
        Thermoviscous absorption coefficient alpha(f) in Np/m.
        alpha(f) = (delta * omega^2) / (2 * c0^3)
        """
        omega = 2.0 * np.pi * freq_hz
        return (self.delta * (omega ** 2)) / (2.0 * (self.c0 ** 3))


@dataclass
class ParametricTransducerConfig:
    """Physical configuration of the ultrasonic parametric emitter."""
    f1_hz: float = 100_000.0    # Carrier frequency 1 (Hz)
    f2_hz: float = 101_000.0    # Carrier frequency 2 (Hz)
    source_radius_m: float = 0.05  # Transducer radius a (m)
    p0_peak_pa: float = 500.0   # Peak surface pressure amplitude (Pa) ~ 148 dB SPL
    modulation_index: float = 1.0  # Modulation depth m in [0, 1]

    @property
    def difference_freq_hz(self) -> float:
        """Difference frequency (audio spotlight demodulation) fd = |f2 - f1|."""
        return abs(self.f2_hz - self.f1_hz)

    @property
    def mean_carrier_freq_hz(self) -> float:
        """Mean carrier frequency fc = (f1 + f2) / 2."""
        return (self.f1_hz + self.f2_hz) / 2.0

    @property
    def source_area_m2(self) -> float:
        """Transducer active radiating area S = pi * a^2."""
        return np.pi * (self.source_radius_m ** 2)

    @property
    def rayleigh_distance_m(self) -> float:
        """Rayleigh nearfield-farfield transition distance z_R = S / lambda_carrier."""
        lambda_c = 343.2 / self.mean_carrier_freq_hz
        return (np.pi * (self.source_radius_m ** 2)) / lambda_c


@dataclass
class WesterveltSimulationResult:
    """Simulation outcomes from the Westervelt solver."""
    z_grid_m: np.ndarray
    t_grid_s: np.ndarray
    p_matrix_pa: np.ndarray  # Shape: (len(z_grid), len(t_grid))
    demodulated_audio_pa: np.ndarray  # Extracted difference-frequency audio signal along z
    audio_spl_db: np.ndarray          # Audio SPL (dB re 20 uPa) along z
    carrier_spl_db: np.ndarray        # Ultrasonic carrier SPL along z
    spectral_frequencies_hz: np.ndarray
    power_spectral_density: np.ndarray # FFT spectrum at far-field end
    beam_angles_rad: Optional[np.ndarray] = None
    beam_directivity_db: Optional[np.ndarray] = None


class WesterveltParametricSolver:
    """
    High-fidelity numerical solver for nonlinear parametric acoustic arrays.
    
    Implements:
    1. 1D retarded-time progressive Westervelt/Burgers finite-difference solver.
    2. Berktay analytical envelope demodulation calculation.
    3. Angular directivity projection for audio spotlighting.
    """

    def __init__(
        self,
        medium: Optional[AirMediumProperties] = None,
        transducer: Optional[ParametricTransducerConfig] = None
    ):
        self.medium = medium or AirMediumProperties()
        self.transducer = transducer or ParametricTransducerConfig()

    def generate_source_signal(self, t: np.ndarray) -> np.ndarray:
        """
        Generates dual-carrier ultrasonic source waveform p(0, t).
        """
        f1 = self.transducer.f1_hz
        f2 = self.transducer.f2_hz
        p0 = self.transducer.p0_peak_pa
        # Dual-frequency superposition: p(t) = 0.5 * p0 * (sin(2*pi*f1*t) + sin(2*pi*f2*t))
        return 0.5 * p0 * (np.sin(2.0 * np.pi * f1 * t) + np.sin(2.0 * np.pi * f2 * t))

    def solve_berktay_demodulation(
        self,
        z_distances: np.ndarray,
        t_duration_s: float = 0.01,
        sampling_rate_hz: float = 1_000_000.0
    ) -> Dict[str, np.ndarray]:
        """
        Berktay's Far-field Parametric Demodulation Analytical Model.
        
        Evaluates demodulated acoustic pressure:
        p_audio(z, t) = [beta * S / (16 * pi * rho0 * c0^4 * alpha_u * z)] * d^2(E(t)^2)/dt^2
        """
        N = int(t_duration_s * sampling_rate_hz)
        t = np.linspace(0.0, t_duration_s, N, endpoint=False)
        dt = t[1] - t[0]

        fc = self.transducer.mean_carrier_freq_hz
        fd = self.transducer.difference_freq_hz
        p0 = self.transducer.p0_peak_pa
        S = self.transducer.source_area_m2
        alpha_u = self.medium.absorption_coefficient(fc)

        # Ultrasonic envelope: E(t) = p0 * cos(pi * fd * t)
        # E(t)^2 = p0^2 * cos^2(pi * fd * t)
        omega_d = 2.0 * np.pi * fd
        e2_t = (p0 * np.cos(np.pi * fd * t)) ** 2
        d2_e2_dt2 = np.gradient(np.gradient(e2_t, dt), dt)

        factor_const = (self.medium.beta * S) / (
            16.0 * np.pi * self.medium.rho0 * (self.medium.c0 ** 4) * max(alpha_u, 1e-6)
        )

        p_audio_at_z = np.zeros((len(z_distances), N))
        audio_spl = np.zeros(len(z_distances))
        pref = 20e-6 # 20 uPa standard reference

        for idx, z in enumerate(z_distances):
            z_eff = max(z, self.transducer.source_radius_m)
            p_audio = (factor_const / z_eff) * d2_e2_dt2
            p_audio_at_z[idx, :] = p_audio
            rms_p = np.sqrt(np.mean(p_audio ** 2))
            audio_spl[idx] = 20.0 * np.log10(max(rms_p, 1e-12) / pref)

        return {
            't_grid': t,
            'z_grid': z_distances,
            'p_audio_matrix': p_audio_at_z,
            'audio_spl_db': audio_spl,
            'difference_freq_hz': np.array([fd])
        }

    def solve_progressive_westervelt_1d(
        self,
        z_max_m: float = 2.0,
        nz: int = 150,
        t_periods: int = 40,
        samples_per_period: int = 32
    ) -> WesterveltSimulationResult:
        """
        Solves 1D Progressive Westervelt / Burgers Equation in Retarded Time Frame:
        tau = t - z/c0
        
        d p / dz = (delta / (2 * c0^3)) * d^2 p / d tau^2 + (beta * p / (rho0 * c0^3)) * d p / d tau
        """
        fc = self.transducer.mean_carrier_freq_hz
        fd = max(self.transducer.difference_freq_hz, 100.0)
        
        # Fundamental carrier period
        T_carrier = 1.0 / fc
        # Compute time window containing integer difference cycles
        T_audio = 1.0 / fd
        dt = T_carrier / samples_per_period
        nt = int(np.round(T_audio * 2 / dt))
        if nt % 2 != 0:
            nt += 1
        t_window = np.arange(nt) * dt
        
        z_grid = np.linspace(0.0, z_max_m, nz)
        dz = z_grid[1] - z_grid[0]
        
        # Initial boundary condition at z = 0
        p_current = self.generate_source_signal(t_window)
        
        p_matrix = np.zeros((nz, nt))
        p_matrix[0, :] = p_current

        # Frequency domain diffusion operator for split-step
        omega = 2.0 * np.pi * np.fft.fftfreq(nt, d=dt)
        diff_coef = self.medium.delta / (2.0 * (self.medium.c0 ** 3))
        absorption_half = np.exp(-diff_coef * (omega ** 2) * (dz / 2.0))
        
        nonlin_coef = self.medium.beta / (self.medium.rho0 * (self.medium.c0 ** 3))

        # Spatial stepping loop (Split-step algorithm)
        for iz in range(1, nz):
            # Step 1: Linear absorption (half-step)
            P_fft = np.fft.fft(p_current) * absorption_half
            p_half = np.real(np.fft.ifft(P_fft))

            # Step 2: Nonlinear steepening (full-step)
            # dp/dz = 0.5 * nonlin_coef * d(p^2)/dtau
            p2 = p_half ** 2
            dp2_dtau = (np.roll(p2, -1) - np.roll(p2, 1)) / (2.0 * dt)
            p_nonlin = p_half + dz * (0.5 * nonlin_coef * dp2_dtau)

            # Step 3: Linear absorption (second half-step)
            P_fft2 = np.fft.fft(p_nonlin) * absorption_half
            p_current = np.real(np.fft.ifft(P_fft2))

            p_matrix[iz, :] = p_current

        # Demodulate audio signal via low-pass filtering at difference frequency cutoff
        nyq = 0.5 / dt
        cutoff = min(fd * 2.5, fc * 0.2)
        norm_cutoff = min(max(cutoff / nyq, 0.001), 0.99)
        b, a = butter(4, norm_cutoff, btype='low')
        demodulated_audio = np.zeros(nt)
        audio_spl = np.zeros(nz)
        carrier_spl = np.zeros(nz)
        pref = 20e-6

        for iz in range(nz):
            sig = p_matrix[iz, :]
            audio_filtered = filtfilt(b, a, sig)
            if iz == nz - 1:
                demodulated_audio = audio_filtered
            rms_aud = np.sqrt(np.mean(audio_filtered ** 2))
            rms_car = np.sqrt(np.mean(sig ** 2))
            audio_spl[iz] = 20.0 * np.log10(max(rms_aud, 1e-12) / pref)
            carrier_spl[iz] = 20.0 * np.log10(max(rms_car, 1e-12) / pref)

        # Spectral analysis at final distance
        final_sig = p_matrix[-1, :]
        fft_vals = np.abs(np.fft.rfft(final_sig)) / nt
        fft_freqs = np.fft.rfftfreq(nt, d=dt)

        # Angular directivity computation (Westervelt virtual end-fire array)
        theta_rad = np.linspace(-np.pi / 2, np.pi / 2, 181)
        k_audio = 2.0 * np.pi * fd / self.medium.c0
        alpha_u = self.medium.absorption_coefficient(fc)
        d_theta = 1.0 / np.sqrt(1.0 + ((2.0 * k_audio / max(alpha_u, 1e-4)) * (np.sin(theta_rad / 2.0) ** 2)) ** 2)
        directivity_db = 20.0 * np.log10(np.maximum(d_theta, 1e-6))

        return WesterveltSimulationResult(
            z_grid_m=z_grid,
            t_grid_s=t_window,
            p_matrix_pa=p_matrix,
            demodulated_audio_pa=demodulated_audio,
            audio_spl_db=audio_spl,
            carrier_spl_db=carrier_spl,
            spectral_frequencies_hz=fft_freqs,
            power_spectral_density=fft_vals,
            beam_angles_rad=theta_rad,
            beam_directivity_db=directivity_db
        )

    def calculate_parametric_beamwidth_deg(self, sim_res: WesterveltSimulationResult) -> float:
        """
        Computes the -3dB half-power full beamwidth of the audio spotlight in degrees.
        """
        if sim_res.beam_angles_rad is None or sim_res.beam_directivity_db is None:
            return 0.0
        
        angles_deg = np.rad2deg(sim_res.beam_angles_rad)
        directivity = sim_res.beam_directivity_db
        max_val = np.max(directivity)
        half_power_mask = directivity >= (max_val - 3.0)
        valid_angles = angles_deg[half_power_mask]
        if len(valid_angles) < 2:
            return 0.0
        return float(np.max(valid_angles) - np.min(valid_angles))
