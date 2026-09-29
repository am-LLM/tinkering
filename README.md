# tinkering

Personal laboratory workbooks, physical simulation kernels, and cross-domain engineering prototypes.

This repository consolidates raw implementations, mathematical derivations, and solver harnesses developed across several years of independent technical investigations. Rather than keeping these scattered across disconnected local scratchpads, they are indexed here as reproducible proof-of-concepts, numerical algorithms, and low-level prototypes.

---

## 🛠️ Philosophy & Engineering Constraints

Most real-world engineering failures occur at the boundaries between disciplines. When problems are viewed strictly through the lens of a single domain, solutions quickly become bloated, fragile, or compute-heavy.

1. **Isomorphic State-Space Invariance**: Physical conservation laws repeat across nature. An acoustic shockwave in a penstock (Joukowsky transient), an electromagnetic reflection on an impedance-mismatched transmission line, and a liquidity cascade in an electronic order book share identical wave equations. Translating problems into their mathematical duals often turns an intractable PDE into an ODE with an analytical solution.
2. **Demoscene-Grade Constraint Optimization**: Maximizing physical fidelity and numerical throughput on constrained hardware (bare-metal ARM Cortex-M, FPGA registers, low-power microcontrollers). Preferring fixed-point arithmetic, vectorized SIMD kernels, and minimal memory footprints over heavyweight runtimes.
3. **Adversarial & Fault-Tolerant Architecture**: Designing under zero-trust assumptions—sensors drift, telemetry gets jammed, 30% of mesh packets drop, and power rails sag under load. Implementing triple-modular redundancy (TMR), chi-square Mahalanobis outlier gating, and SMT (Z3) formal verification to mathematically guarantee bounded execution.

---

## 📂 Repository Layout

```text
.
├── frontier_hybrids/          # 65 Cross-domain Frankenstein engines (Biomimetic, Quantum, GNC, PQC)
│   ├── engine_01...engine_65  # Standalone numerical physics & control implementations
│   └── tests/                 # Full pytest harness (100% empirical pass rate)
│
├── domain_laboratories/       # Specialized multi-disciplinary testbenches
│   ├── aerospace_gnc/         # 15-state ES-EKF, ULA MVDR beamforming, Space Shuttle TMR FDIR, Z3 proofs
│   ├── frugal_mechanics/      # Method of Characteristics (MOC) water-hammer, Seebeck MPPT, 2-RC ECM BMS
│   ├── advanced_quantum_scada/# Tokamak MHD equilibrium, QKD satellite links, Modbus deep packet firewalls
│   ├── isomorphic_hybrid/     # Bidirectional state-space physics bridge, PQC Ring-LWE consensus
│   ├── acoustic_metamaterials/# Westervelt non-linear acoustics, lithospheric rate-state friction
│   ├── c_baremetal_quant/     # Zero-allocation C limit order book and lock-free ringbuffers
│   └── ...                    # Autonomous drone meshes, extremophile genetics, OSINT DAGs
│
├── engineering_continuum/     # 418 Modular domain harnesses (Courses 001 - 418)
│   ├── course_001_...         # Verified implementations spanning electromagnetics, RF, robotics,
│   └── course_418_...         # nuclear point kinetics, control barrier functions, and neural ODEs
│
├── verify_all.py              # Master empirical verification test suite
└── requirements.txt           # Minimal Python scientific dependencies (numpy, scipy, sympy, z3-solver)
```

---

## 🔬 Featured Frontier Hybrid Engines

| Engine | Domains Unified | Core Physical / Mathematical Mechanism |
| :--- | :--- | :--- |
| **Engine 01** | Immunology + Swarm UAVs | Affinity maturation clonal selection for Byzantine GPS-denied rendezvous. |
| **Engine 04** | PQC + SCADA Power Grids | Schnorr-Pedersen NIZK telemetry proofs with ring-LWE lattice key encapsulation. |
| **Engine 10** | Astrophysics + Sonar | Matched filter gravitational-wave chirp extraction for low-SNR subsea acoustic steganography. |
| **Engine 18** | Relativistic Plasma + Opt | 1D PIC plasma wakefield acceleration with laser ponderomotive envelope solvers. |
| **Engine 26** | Magnetohydrodynamics | Hartmann flow molten-salt electromagnetic pump using Navier-Stokes Lorentz coupling. |
| **Engine 51** | Quantum Radar + Chirp | Hyperbolic chirp matched-filter radar resolving targets at negative SNR ($SNR = -15\text{ dB}$). |
| **Engine 57** | Geothermal + Squeezed DAS| Quantum-squeezed Distributed Acoustic Sensing for micro-fracture localization. |
| **Engine 60** | Astrocytic Tripartite SNN | Astrocyte calcium dynamics modulating spike-timing-dependent plasticity (STDP). |
| **Engine 64** | Non-Hermitian Physics + Med | Exceptional Point (EP) second-order eigenvalue splitting for ultra-early sepsis detection. |
| **Engine 65** | Trajectory + Extreme Value | Fréchet EVT tail-risk trajectory optimizer navigating non-Gaussian EW RF jammers. |

---

## ⚡ Quickstart & Verification

Ensure you have Python 3.10+ installed with standard numerical tooling:

```bash
git clone https://github.com/am-LLM/tinkering.git
cd tinkering
pip install -r requirements.txt
```

Run the master empirical test harness:

```bash
python3 verify_all.py
```

All 65 frontier hybrid engines, domain laboratories, and continuum testbenches execute locally and verify zero runtime errors and numerical convergence.

---

## 📜 Author

**Ali Malik** ([@am-LLM](https://github.com/am-LLM))  
*Cross-domain systems architecture, low-resource hardware/software engineering, and physics-informed computational modeling.*
