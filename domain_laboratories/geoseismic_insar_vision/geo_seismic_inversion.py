"""
2D Acoustic Full Waveform Inversion (FWI) Engine.

Solves 2D subterranean seismic wave propagation via high-order Finite-Difference
Time-Domain (FDTD) and reconstructs underground velocity models using
the Adjoint-State back-propagation method.

Theoretical Foundations:
-----------------------
1. Acoustic Wave Equation:
   (1 / v(x,z)^2) * d^2 p / dt^2 = d^2 p / dx^2 + d^2 p / dz^2 + s(x, z, t)

2. FDTD Discretization:
   - 4th-order spatial Laplacian + 2nd-order leapfrog time stepping.
   - Cerjan absorbing sponge boundary layer to eliminate artificial edge reflections.

3. Objective Function:
   J(v) = 0.5 * sum_{shots} sum_{receivers} int_0^T [ p_syn(x_r, z_r, t) - p_obs(x_r, z_r, t) ]^2 dt

4. Adjoint-State Gradient:
   g(x, z) = - (2 / v(x,z)^3) * int_0^T [ d^2 p / dt^2 (x, z, t) * lambda(x, z, t) ] dt
   where lambda(x, z, t) is the adjoint wavefield back-propagated from receiver residuals.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Tuple


@dataclass
class SeismicGridConfig:
    """Spatial and temporal grid parameters for 2D FDTD simulation."""
    nx: int = 80               # Grid points in X (horizontal)
    nz: int = 60               # Grid points in Z (depth)
    dx_m: float = 10.0         # Spatial spacing dx (m)
    dz_m: float = 10.0         # Spatial spacing dz (m)
    dt_s: float = 0.001        # Time step dt (s) (must satisfy CFL condition)
    nt: int = 400              # Number of time steps
    nboundary: int = 15        # Absorbing sponge layer thickness in grid cells
    damping_factor: float = 0.015 # Cerjan exponential damping coefficient

    @property
    def total_nx(self) -> int:
        return self.nx + 2 * self.nboundary

    @property
    def total_nz(self) -> int:
        return self.nz + 2 * self.nboundary

    @property
    def x_coords_m(self) -> np.ndarray:
        return np.arange(self.nx) * self.dx_m

    @property
    def z_coords_m(self) -> np.ndarray:
        return np.arange(self.nz) * self.dz_m


@dataclass
class SeismicSurveyGeometry:
    """Multi-shot and multi-receiver acquisition geometry."""
    shot_x_indices: List[int]      # Source grid indices in X
    shot_z_indices: List[int]      # Source grid indices in Z
    receiver_x_indices: List[int]  # Receiver grid indices in X
    receiver_z_indices: List[int]  # Receiver grid indices in Z
    ricker_f0_hz: float = 15.0     # Source peak frequency (Hz)
    t0_s: float = 0.05             # Source time delay (s)

    def generate_ricker_wavelet(self, nt: int, dt: float) -> np.ndarray:
        """Generates zero-phase Ricker wavelet source signature."""
        t = np.arange(nt) * dt - self.t0_s
        tau = np.pi * self.ricker_f0_hz * t
        return (1.0 - 2.0 * (tau ** 2)) * np.exp(- (tau ** 2))


@dataclass
class FWIInversionResult:
    """Output of Full Waveform Inversion optimization."""
    initial_velocity_mps: np.ndarray
    inverted_velocity_mps: np.ndarray
    true_velocity_mps: Optional[np.ndarray]
    cost_history: List[float]
    final_gradient: np.ndarray
    iterations_completed: int


class FullWaveformInversion2D:
    """
    2D Acoustic Full Waveform Inversion (FWI) Engine.
    """

    def __init__(self, grid: SeismicGridConfig, geometry: SeismicSurveyGeometry):
        self.grid = grid
        self.geom = geometry
        self._build_absorbing_boundary()

    def _build_absorbing_boundary(self):
        """Creates 2D Cerjan exponential damping mask."""
        total_nx = self.grid.total_nx
        total_nz = self.grid.total_nz
        nb = self.grid.nboundary
        damp = self.grid.damping_factor

        self.sponge_mask = np.ones((total_nz, total_nx), dtype=np.float64)
        for iz in range(total_nz):
            for ix in range(total_nx):
                dist_x = 0
                dist_z = 0
                if ix < nb:
                    dist_x = nb - ix
                elif ix >= total_nx - nb:
                    dist_x = ix - (total_nx - nb - 1)
                
                if iz < nb:
                    dist_z = nb - iz
                elif iz >= total_nz - nb:
                    dist_z = iz - (total_nz - nb - 1)

                dist = max(dist_x, dist_z)
                if dist > 0:
                    self.sponge_mask[iz, ix] = np.exp(- (damp * dist) ** 2)

    def _pad_velocity(self, v_core: np.ndarray) -> np.ndarray:
        """Pads core velocity model with boundary layer."""
        nb = self.grid.nboundary
        return np.pad(v_core, ((nb, nb), (nb, nb)), mode='edge')

    def _laplacian_4th_order(self, p: np.ndarray) -> np.ndarray:
        """
        Computes 4th-order central finite difference Laplacian:
        d^2 p / dx^2 + d^2 p / dz^2
        """
        dx = self.grid.dx_m
        dz = self.grid.dz_m
        dx2 = dx ** 2
        dz2 = dz ** 2

        # Coefficients for 4th order 1D 2nd derivative: [-1/12, 4/3, -5/2, 4/3, -1/12]
        c0 = -2.5
        c1 = 4.0 / 3.0
        c2 = -1.0 / 12.0

        # X-derivative
        d2p_dx2 = (
            c2 * (np.roll(p, -2, axis=1) + np.roll(p, 2, axis=1))
            + c1 * (np.roll(p, -1, axis=1) + np.roll(p, 1, axis=1))
            + c0 * p
        ) / dx2

        # Z-derivative
        d2p_dz2 = (
            c2 * (np.roll(p, -2, axis=0) + np.roll(p, 2, axis=0))
            + c1 * (np.roll(p, -1, axis=0) + np.roll(p, 1, axis=0))
            + c0 * p
        ) / dz2

        return d2p_dx2 + d2p_dz2

    def forward_propagate(
        self,
        velocity_mps: np.ndarray,
        shot_idx: int = 0,
        save_wavefield: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward time stepping of acoustic wave equation.
        Returns:
            (receiver_seismograms, full_wavefield_history)
        """
        v_padded = self._pad_velocity(velocity_mps)
        v2_dt2 = (v_padded ** 2) * (self.grid.dt_s ** 2)

        total_nx = self.grid.total_nx
        total_nz = self.grid.total_nz
        nt = self.grid.nt
        dt = self.grid.dt_s
        nb = self.grid.nboundary

        p_prev = np.zeros((total_nz, total_nx), dtype=np.float64)
        p_curr = np.zeros((total_nz, total_nx), dtype=np.float64)
        p_next = np.zeros((total_nz, total_nx), dtype=np.float64)

        src_wavelet = self.geom.generate_ricker_wavelet(nt, dt)
        sx = self.geom.shot_x_indices[shot_idx] + nb
        sz = self.geom.shot_z_indices[shot_idx] + nb

        num_recs = len(self.geom.receiver_x_indices)
        seismograms = np.zeros((num_recs, nt), dtype=np.float64)
        
        wavefield_hist = np.zeros((nt, total_nz, total_nx), dtype=np.float32) if save_wavefield else None

        for it in range(nt):
            # Compute spatial Laplacian
            lap = self._laplacian_4th_order(p_curr)

            # Time stepping: p_next = 2*p_curr - p_prev + v^2*dt^2 * lap
            p_next = 2.0 * p_curr - p_prev + v2_dt2 * lap

            # Inject source term
            p_next[sz, sx] += (v2_dt2[sz, sx] / (self.grid.dx_m * self.grid.dz_m)) * src_wavelet[it]

            # Apply absorbing sponge layer
            p_next *= self.sponge_mask
            p_curr *= self.sponge_mask

            # Record receivers
            for ir, (rx, rz) in enumerate(zip(self.geom.receiver_x_indices, self.geom.receiver_z_indices)):
                seismograms[ir, it] = p_next[rz + nb, rx + nb]

            if save_wavefield:
                wavefield_hist[it, :, :] = p_curr

            # Update time slices
            p_prev = p_curr.copy()
            p_curr = p_next.copy()

        return seismograms, wavefield_hist

    def compute_adjoint_gradient(
        self,
        velocity_mps: np.ndarray,
        observed_seismograms_by_shot: List[np.ndarray]
    ) -> Tuple[float, np.ndarray]:
        """
        Computes FWI objective cost and adjoint gradient:
        g(x, z) = - (2 / v^3) * int d2p/dt2 * lambda dt
        """
        total_cost = 0.0
        nb = self.grid.nboundary
        nx = self.grid.nx
        nz = self.grid.nz
        nt = self.grid.nt
        dt = self.grid.dt_s

        v_padded = self._pad_velocity(velocity_mps)
        v2_dt2 = (v_padded ** 2) * (dt ** 2)
        total_gradient_padded = np.zeros((self.grid.total_nz, self.grid.total_nx), dtype=np.float64)

        for shot_idx in range(len(self.geom.shot_x_indices)):
            # 1. Forward run saving wavefield
            syn_seis, forward_wavefield = self.forward_propagate(velocity_mps, shot_idx=shot_idx, save_wavefield=True)
            obs_seis = observed_seismograms_by_shot[shot_idx]

            # 2. Compute residual seismograms (syn - obs)
            residuals = syn_seis - obs_seis
            shot_cost = 0.5 * np.sum(residuals ** 2) * dt
            total_cost += float(shot_cost)

            # 3. Backward propagation of adjoint wavefield
            adj_prev = np.zeros((self.grid.total_nz, self.grid.total_nx), dtype=np.float64)
            adj_curr = np.zeros((self.grid.total_nz, self.grid.total_nx), dtype=np.float64)
            adj_next = np.zeros((self.grid.total_nz, self.grid.total_nx), dtype=np.float64)

            for it in range(nt - 1, -1, -1):
                lap_adj = self._laplacian_4th_order(adj_curr)
                adj_next = 2.0 * adj_curr - adj_prev + v2_dt2 * lap_adj

                # Inject adjoint residual sources at receiver locations
                for ir, (rx, rz) in enumerate(zip(self.geom.receiver_x_indices, self.geom.receiver_z_indices)):
                    adj_next[rz + nb, rx + nb] += (v2_dt2[rz + nb, rx + nb] / (self.grid.dx_m * self.grid.dz_m)) * residuals[ir, it]

                adj_next *= self.sponge_mask
                adj_curr *= self.sponge_mask

                # Compute acceleration d2p/dt2 of forward wavefield
                if 1 <= it < nt - 1:
                    d2p_dt2 = (forward_wavefield[it+1] - 2.0 * forward_wavefield[it] + forward_wavefield[it-1]) / (dt ** 2)
                    total_gradient_padded += - (2.0 / (v_padded ** 3)) * d2p_dt2 * adj_curr * dt

                adj_prev = adj_curr.copy()
                adj_curr = adj_next.copy()

        # Crop gradient to active core region
        core_gradient = total_gradient_padded[nb:nb+nz, nb:nb+nx]
        return total_cost, core_gradient

    def invert(
        self,
        v_initial: np.ndarray,
        observed_seis_by_shot: List[np.ndarray],
        max_iterations: int = 5,
        learning_rate_scale: float = 50.0,
        v_min: float = 1200.0,
        v_max: float = 4500.0,
        true_velocity: Optional[np.ndarray] = None
    ) -> FWIInversionResult:
        """
        Runs iterative gradient-descent Full Waveform Inversion.
        """
        v_current = v_initial.copy().astype(np.float64)
        cost_history: List[float] = []
        final_grad = np.zeros_like(v_current)

        for iter_num in range(max_iterations):
            cost, grad = self.compute_adjoint_gradient(v_current, observed_seis_by_shot)
            cost_history.append(cost)
            final_grad = grad.copy()

            # Normalize gradient
            max_abs_g = np.max(np.abs(grad))
            if max_abs_g < 1e-15:
                break
            norm_grad = grad / max_abs_g

            # Parameter update
            v_current = v_current - learning_rate_scale * norm_grad
            # Project onto physical velocity bounds
            v_current = np.clip(v_current, v_min, v_max)

        return FWIInversionResult(
            initial_velocity_mps=v_initial,
            inverted_velocity_mps=v_current,
            true_velocity_mps=true_velocity,
            cost_history=cost_history,
            final_gradient=final_grad,
            iterations_completed=len(cost_history)
        )
