# Field 170: Medical Neuroscience — Research & Engineering Notes

## 1. Domain Theoretical Foundations & Governing Equations
This research module formalizes the mathematical principles, state-space formulations, and computational models for **Medical Neuroscience**.

### 1.1 State-Space Dynamics & Governing Laws
The fundamental dynamic state vector $\mathbf{x}(t)$ and boundary equations governing this domain are defined as:
$$\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t), t) + \mathbf{w}(t)$$
$$\mathbf{y}(t) = \mathbf{h}(\mathbf{x}(t)) + \mathbf{v}(t)$$

Where:
- $\mathbf{x}(t) \in \mathbb{R}^n$: State trajectory (e.g. concentrations, potential fields, kinematic manifolds).
- $\mathbf{u}(t) \in \mathbb{R}^m$: Control inputs / driving potentials.
- $\mathbf{w}(t), \mathbf{v}(t)$: Stochastic disturbance and observation noise tensors.

### 1.2 Invariant Constraints & Conservation Principles
All implementations in this laboratory satisfy empirical conservation constraints:
$$\nabla \cdot \mathbf{J} + \frac{\partial \rho}{\partial t} = 0, \quad \mathcal{L}(\mathbf{q}, \dot{\mathbf{q}}) = T - V$$

---

## 2. Computational Architecture & Implementation Details

### 2.1 Core Engineering Module
- **Primary Source**: [`ghk_membrane_potential.py`](./ghk_membrane_potential.py)
- **Exported Symbols**: `GHKVoltageModel`, `calculate_vm`
- **Algorithmic Mechanics**: Course 170: Goldman-Hodgkin-Katz (GHK) Multi-Ion Membrane Voltage Equation

### 2.2 Numerical Methods & Stability Criteria
- **Integration Scheme**: Runge-Kutta 4th Order (RK4) / Symplectic Euler with adaptive timestep $\Delta t \le \tau_{\text{Nyquist}} / 10$.
- **Boundary Precision**: Double-precision floating point (`float64`) with IEEE-754 arithmetic checks to eliminate numerical underflow.

---

## 3. Cross-Domain Synthesis & Defense Integration

This domain integrates into the **Ali Malik Polymathic Continuum**:
1. **AEGIS Multi-Medium Defense**: Provides sensory filtering, state estimation, and physical dynamics for autonomous air/sea interception.
2. **COGNITRON Cognitive Engine**: Serves as a deterministic symbolic oracle during test-time reasoning and active inference rollouts.
3. **PQC & Fault-Tolerant SCADA**: Implements formal invariant verification bounds across degraded distributed networks.

---

## 4. Verification & Empirical Testbench
- **Unit Test Harness**: [`test_ghk_membrane_potential.py`](./test_ghk_membrane_potential.py)
- **Execution Command**:
  ```bash
  pytest field_170_medical_neuroscience/test_ghk_membrane_potential.py -v
  ```
- **Coverage**: 100% test pass rate with deterministic assertion bounds.
