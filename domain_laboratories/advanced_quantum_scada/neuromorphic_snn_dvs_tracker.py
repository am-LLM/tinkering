"""
Neuromorphic Leaky Integrate-and-Fire (LIF) Spiking Neural Network (SNN) Event Camera Processor.

Features:
- Dynamic Vision Sensor (DVS) asynchronous microsecond event stream processing (x, y, t_us, polarity).
- 2D grid LIF neurons with membrane leakage, refractory periods, and threshold spiking.
- Spike-Timing-Dependent Plasticity (STDP) unsupervised synaptic weight learning.
- Microsecond Time-Surface Exponential Decay & Optical Flow Velocity Vector Estimation.
- Real-time bounding box target tracking & trajectory estimation.
- Neuromorphic hardware power benchmarking (<10 mW ultra-low power consumption model).
"""

from dataclasses import dataclass, field
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class DVSEvent:
    """Microsecond Dynamic Vision Sensor (DVS) asynchronous event."""
    x: int             # Pixel X coordinate
    y: int             # Pixel Y coordinate
    timestamp_us: int  # Microsecond timestamp
    polarity: int      # +1 (ON event / increase) or -1 (OFF event / decrease)


@dataclass
class LIFNeuronConfig:
    """Leaky Integrate-and-Fire neuron parameters."""
    tau_m_ms: float = 20.0       # Membrane time constant tau_m (ms)
    v_rest: float = -70.0        # Resting potential (mV)
    v_reset: float = -75.0       # Reset potential (mV)
    v_threshold: float = -55.0   # Spiking threshold (mV)
    refractory_period_us: int = 1000 # Absolute refractory period (1000 us = 1 ms)
    resistance_mohm: float = 10.0 # Membrane resistance R_m (MOhm)


@dataclass
class STDPConfig:
    """Spike-Timing-Dependent Plasticity learning parameters."""
    a_plus: float = 0.015        # Potentiation learning rate
    a_minus: float = 0.012       # Depression learning rate
    tau_plus_ms: float = 20.0    # Potentiation time constant (ms)
    tau_minus_ms: float = 20.0   # Depression time constant (ms)
    w_min: float = 0.0           # Minimum synaptic weight
    w_max: float = 1.0           # Maximum synaptic weight


@dataclass
class NeuromorphicPowerConfig:
    """Hardware energy specifications for neuromorphic ASIC (e.g. Loihi / TrueNorth / DynapSE)."""
    static_leakage_power_mw: float = 1.2 # Baseline standby leakage power in mW
    energy_per_sop_pj: float = 1.5       # Energy per Synaptic Operation in picoJoules (pJ)
    energy_per_neuron_spike_pj: float = 4.0 # Energy per soma spike in picoJoules (pJ)


class DVSStreamGenerator:
    """Generates synthetic DVS event streams for moving objects across sensor field."""

    @staticmethod
    def generate_moving_target(
        width: int = 64,
        height: int = 64,
        duration_us: int = 100000,   # 100 ms simulation
        velocity_px_per_s: Tuple[float, float] = (150.0, 100.0), # (vx, vy) in px/s
        target_radius: float = 3.0,
        noise_event_rate_hz: float = 50.0,
        seed: Optional[int] = None
    ) -> List[DVSEvent]:
        """Synthesize DVS events for a moving target plus Poisson background noise."""
        rng = np.random.default_rng(seed)
        events: List[DVSEvent] = []

        dt_step_us = 200 # 200 us simulation time resolution
        vx_px_us = velocity_px_per_s[0] / 1.0e6
        vy_px_us = velocity_px_per_s[1] / 1.0e6

        start_x = width * 0.2
        start_y = height * 0.2

        curr_t = 0
        while curr_t < duration_us:
            cx = start_x + vx_px_us * curr_t
            cy = start_y + vy_px_us * curr_t

            # Generate edge events for the circle perimeter
            if 0 <= cx < width and 0 <= cy < height:
                for angle in np.linspace(0, 2 * math.pi, num=8, endpoint=False):
                    ex = int(round(cx + target_radius * math.cos(angle)))
                    ey = int(round(cy + target_radius * math.sin(angle)))
                    if 0 <= ex < width and 0 <= ey < height:
                        pol = 1 if math.sin(angle + curr_t * 1e-4) > 0 else -1
                        jitter = int(rng.integers(-50, 50))
                        events.append(DVSEvent(ex, ey, max(0, curr_t + jitter), pol))

            # Add Poisson thermal noise
            num_noise = rng.poisson(noise_event_rate_hz * (dt_step_us * 1e-6) * (width * height / 100.0))
            for _ in range(num_noise):
                nx = int(rng.integers(0, width))
                ny = int(rng.integers(0, height))
                events.append(DVSEvent(nx, ny, curr_t, int(rng.choice([-1, 1]))))

            curr_t += dt_step_us

        # Sort chronologically
        events.sort(key=lambda e: e.timestamp_us)
        return events


class LIFSpikingLayer:
    """2D Grid of Leaky Integrate-and-Fire Neurons."""

    def __init__(self, width: int, height: int, config: Optional[LIFNeuronConfig] = None):
        self.width = width
        self.height = height
        self.config = config or LIFNeuronConfig()

        # State tensors
        self.v_membrane = np.full((height, width), self.config.v_rest, dtype=np.float64)
        self.last_update_t_us = np.zeros((height, width), dtype=np.int64)
        self.last_spike_t_us = np.full((height, width), -1000000, dtype=np.int64)
        self.spike_count = 0

    def decay_membrane(self, y: int, x: int, current_t_us: int):
        """Exponential decay: V(t) = V_rest + (V_prev - V_rest) * exp(-dt / tau_m)."""
        dt_ms = (current_t_us - self.last_update_t_us[y, x]) / 1000.0
        if dt_ms > 0:
            decay_factor = math.exp(-dt_ms / self.config.tau_m_ms)
            self.v_membrane[y, x] = self.config.v_rest + (self.v_membrane[y, x] - self.config.v_rest) * decay_factor
            self.last_update_t_us[y, x] = current_t_us

    def inject_current_and_update(self, y: int, x: int, current_t_us: int, weight: float) -> bool:
        """
        Inject synaptic current into neuron (y, x).
        Returns True if neuron fires an output action potential (spike).
        """
        if current_t_us - self.last_spike_t_us[y, x] < self.config.refractory_period_us:
            # In refractory period
            return False

        self.decay_membrane(y, x, current_t_us)
        # Membrane integration: delta_V = R_m * I_syn
        delta_v = weight * 5.0  # 5 mV per unit synaptic weight
        self.v_membrane[y, x] += delta_v

        # Threshold check
        if self.v_membrane[y, x] >= self.config.v_threshold:
            self.v_membrane[y, x] = self.config.v_reset
            self.last_spike_t_us[y, x] = current_t_us
            self.spike_count += 1
            return True
        return False


class NeuromorphicTracker:
    """
    Neuromorphic Target Tracker combining LIF SNN, STDP Learning,
    Time-Surface Flow estimation, and ultra-low power metrics.
    """

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        tau_surface_us: float = 20000.0, # 20 ms decay for time-surface
        lif_config: Optional[LIFNeuronConfig] = None,
        stdp_config: Optional[STDPConfig] = None,
        power_config: Optional[NeuromorphicPowerConfig] = None
    ):
        self.width = width
        self.height = height
        self.tau_surface_us = tau_surface_us
        self.lif_layer = LIFSpikingLayer(width, height, lif_config)
        self.stdp = stdp_config or STDPConfig()
        self.power_cfg = power_config or NeuromorphicPowerConfig()

        # Time-surface: last event timestamp per pixel
        self.last_event_time = np.full((height, width), -1.0, dtype=np.float64)
        # Synaptic weights connecting input pixels to LIF neurons (locally connected 3x3 kernels)
        self.weights = np.ones((height, width, 3, 3), dtype=np.float64) * 0.5
        self.total_sops = 0
        self.tracked_centroids: List[Tuple[float, float, int]] = [] # (x, y, timestamp_us)

    def compute_time_surface(self, y: int, x: int, current_t_us: int, radius: int = 2) -> np.ndarray:
        """
        Compute local exponential time surface S_e(x, y) = exp(-(t_curr - t_last) / tau).
        """
        y_min, y_max = max(0, y - radius), min(self.height, y + radius + 1)
        x_min, x_max = max(0, x - radius), min(self.width, x + radius + 1)

        t_patch = self.last_event_time[y_min:y_max, x_min:x_max]
        dt = current_t_us - t_patch
        # Valid pixels where events occurred
        valid_mask = (t_patch >= 0)
        surface = np.zeros_like(dt)
        surface[valid_mask] = np.exp(-dt[valid_mask] / self.tau_surface_us)
        return surface

    def estimate_optical_flow(self, y: int, x: int, current_t_us: int) -> Tuple[float, float]:
        """
        Estimate local microsecond optical flow velocity vector (vx, vy) in px/s
        using the gradient of the exponential time surface: v = - (grad T) / ||grad T||^2.
        """
        if y <= 1 or y >= self.height - 2 or x <= 1 or x >= self.width - 2:
            return 0.0, 0.0

        surface = self.compute_time_surface(y, x, current_t_us, radius=1)
        if surface.shape != (3, 3) or np.count_nonzero(surface) < 3:
            return 0.0, 0.0

        # Spatial gradients using Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

        gx = np.sum(surface * sobel_x) / 8.0
        gy = np.sum(surface * sobel_y) / 8.0
        grad_norm_sq = gx ** 2 + gy ** 2

        if grad_norm_sq < 1e-6:
            return 0.0, 0.0

        # Velocity in pixels per microsecond -> scale to pixels/sec (1e6)
        # Optical flow constraint: v = -(grad_T / ||grad_T||^2) * (1 / tau)
        scale = 1.0e6 / (self.tau_surface_us * grad_norm_sq)
        vx = float(gx * scale)
        vy = float(gy * scale)
        return vx, vy

    def apply_stdp(self, y: int, x: int, pre_spike_t_us: int, post_spike_t_us: int):
        """
        Apply biological STDP rule:
        delta_w = A_+ * exp(-dt / tau_+) if dt > 0 (post after pre -> potentiation)
        delta_w = -A_- * exp(dt / tau_-) if dt < 0 (pre after post -> depression)
        """
        dt_ms = (post_spike_t_us - pre_spike_t_us) / 1000.0
        if dt_ms > 0:
            dw = self.stdp.a_plus * math.exp(-dt_ms / self.stdp.tau_plus_ms)
        elif dt_ms < 0:
            dw = -self.stdp.a_minus * math.exp(dt_ms / self.stdp.tau_minus_ms)
        else:
            dw = 0.0

        # Update local 3x3 synaptic kernel
        self.weights[y, x] = np.clip(self.weights[y, x] + dw, self.stdp.w_min, self.stdp.w_max)

    def process_event_stream(self, events: List[DVSEvent]) -> Dict[str, Any]:
        """
        Process incoming microsecond DVS event stream through SNN pipeline.
        Tracks target centroid and measures neuromorphic hardware power.
        """
        if not events:
            return {
                'processed_events': 0,
                'soma_spikes': 0,
                'total_sops': 0,
                'power_mw': self.power_cfg.static_leakage_power_mw,
                'target_velocity': (0.0, 0.0),
                'tracked_positions': []
            }

        start_t = events[0].timestamp_us
        end_t = events[-1].timestamp_us
        duration_s = max(1e-6, (end_t - start_t) / 1.0e6)

        recent_spikes: List[Tuple[int, int, int]] = [] # (x, y, t)
        flow_vectors: List[Tuple[float, float]] = []

        for ev in events:
            x, y, t = ev.x, ev.y, ev.timestamp_us
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue

            self.last_event_time[y, x] = float(t)

            # Local receptive field synaptic integration
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        w = self.weights[ny, nx, dy + 1, dx + 1]
                        self.total_sops += 1
                        spiked = self.lif_layer.inject_current_and_update(ny, nx, t, w)
                        if spiked:
                            recent_spikes.append((nx, ny, t))
                            # Apply STDP
                            self.apply_stdp(ny, nx, pre_spike_t_us=t, post_spike_t_us=t)

            # Estimate optical flow periodically
            if len(recent_spikes) > 0 and len(recent_spikes) % 10 == 0:
                vx, vy = self.estimate_optical_flow(y, x, t)
                if abs(vx) > 1e-3 or abs(vy) > 1e-3:
                    flow_vectors.append((vx, vy))

            # Centroid tracking window (~2ms temporal clustering)
            if len(recent_spikes) >= 15:
                window_spikes = recent_spikes[-15:]
                avg_x = float(np.mean([s[0] for s in window_spikes]))
                avg_y = float(np.mean([s[1] for s in window_spikes]))
                self.tracked_centroids.append((avg_x, avg_y, t))

        # Power calculation
        # P_dynamic = (SOP_count * E_SOP + Spike_count * E_spike) / duration
        dynamic_energy_j = (
            (self.total_sops * self.power_cfg.energy_per_sop_pj * 1e-12) +
            (self.lif_layer.spike_count * self.power_cfg.energy_per_neuron_spike_pj * 1e-12)
        )
        p_dynamic_w = dynamic_energy_j / duration_s
        p_dynamic_mw = p_dynamic_w * 1000.0
        p_total_mw = self.power_cfg.static_leakage_power_mw + p_dynamic_mw

        avg_vx = float(np.median([v[0] for v in flow_vectors])) if flow_vectors else 0.0
        avg_vy = float(np.median([v[1] for v in flow_vectors])) if flow_vectors else 0.0

        return {
            'processed_events': len(events),
            'soma_spikes': self.lif_layer.spike_count,
            'total_sops': self.total_sops,
            'dynamic_power_mw': p_dynamic_mw,
            'static_power_mw': self.power_cfg.static_leakage_power_mw,
            'total_power_mw': p_total_mw,
            'estimated_velocity_px_s': (avg_vx, avg_vy),
            'tracked_positions_count': len(self.tracked_centroids),
            'power_under_10mw': (p_total_mw < 10.0)
        }
