# System Architecture & Mathematical Foundations

This document details the governing equations, numerical methods, and constraint optimizations utilized across the engines in this repository.

---

## 1. Mathematical Frameworks & Solvers

### 1.1 Error-State Extended Kalman Filter (ES-EKF)
In `aerospace_gnc/ghost_gnc_vio.py`, orientation error is parameterized on the Lie group $SO(3)$ using error quaternions $\delta \boldsymbol{\theta} \in \mathbb{R}^3$:
$$\mathbf{q} = \hat{\mathbf{q}} \otimes \begin{bmatrix} 1 \\ \frac{1}{2} \delta \boldsymbol{\theta} \end{bmatrix}$$
The 15-state vector tracks nominal position $\mathbf{p}$, velocity $\mathbf{v}$, quaternion $\mathbf{q}$, accelerometer bias $\mathbf{a}_b$, and gyroscope bias $\boldsymbol{\omega}_b$. To prevent spoofed optical flow features from corrupting state estimates, updates are gated using the Mahalanobis distance metric:
$$d_M^2 = \mathbf{r}^T (\mathbf{H} \mathbf{P} \mathbf{H}^T + \mathbf{R})^{-1} \mathbf{r} < \chi_{k, 0.95}^2$$

### 1.2 1D Acoustic Water-Hammer (Method of Characteristics)
In `frugal_mechanics/hydraulic_ram_pump_transient.py`, the hyperbolic PDEs governing unsteady pipe flow (continuity and momentum) are transformed into total differential equations along characteristic lines $C^+$ and $C^-$:
$$\frac{dH}{dt} \pm \frac{a}{g A} \frac{dQ}{dt} + \frac{f a Q |Q|}{2 g D A^2} = 0 \quad \text{along} \quad \frac{dx}{dt} = \pm a$$
where $a$ is wave propagation speed, $D$ is pipe diameter, $f$ is Darcy-Weisbach friction factor, and Courant condition $C = \frac{a \Delta t}{\Delta x} \le 1.0$ is strictly enforced for numerical stability.

### 1.3 Non-Hermitian Exceptional Point (EP) Sensing
In `frontier_hybrids/engine_64_non_hermitian_ep_sepsis_detector.py`, coupled multi-modal biomarker dynamics are represented as an effective non-Hermitian Hamiltonian $\mathcal{H}_{eff} \in \mathbb{C}^{2 \times 2}$ satisfying $\mathcal{P}\mathcal{T}$-symmetry:
$$\mathcal{H}_{eff} = \begin{bmatrix} \omega_0 + i\gamma_1 & \kappa \\ \kappa & \omega_0 - i\gamma_2 \end{bmatrix}$$
At the second-order exceptional point ($\kappa = |\gamma_1 + \gamma_2| / 2$), the eigenvalue splitting scales sub-linearly with external biomarker perturbation $\epsilon$:
$$\Delta \lambda = \lambda_+ - \lambda_- \propto \sqrt{\epsilon}$$
yielding orders-of-magnitude higher sensitivity to minute micro-vascular inflammatory cascades than conventional Hermitian sensors.

---

## 2. Low-Resource & Demoscene Engineering Strategies

1. **Zero-Allocation Execution Loops**:
   - Time-critical physics update loops (`step()`, `integrate()`) pre-allocate state buffers and covariance matrices to prevent heap churn and garbage collector pauses.
2. **Taylor Series & Polynomial Approximations**:
   - High-frequency trigonometric and exponential calls inside micro-stepping controllers use truncated minimax polynomials to conserve CPU clock cycles on bare-metal architectures.
3. **Sparse Matrix & Vectorized SIMD Layouts**:
   - FDTD Yee grids (`course_391`) and finite-difference heat equations store electromagnetic/thermal fields as contiguous 1D/2D arrays, maximizing L1 cache locality and compiler auto-vectorization.

---

## 3. Formal Verification & Safety Boundaries

- **Z3 SMT Invariant Verification**: `aerospace_gnc/z3_formal_swarm_deconflict.py` uses first-order logic and quantifier elimination to prove that minimum Euclidean separation between any two agents $\Delta \mathbf{x}_{ij} \ge d_{safe}$ is invariant across all reachable control state transitions.
- **Lyapunov Asymptotic Stability**: Nonlinear control algorithms verify $\dot{V}(\mathbf{x}) \le -\alpha V(\mathbf{x})$ over state trajectories to guarantee bounded convergence during actuator saturation.
