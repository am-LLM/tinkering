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
├── frontier_hybrids/          # 66 Cross-domain Frankenstein engines (Biomimetic, Quantum, GNC, PQC, Cyber)
│   ├── engine_01...engine_66  # Standalone numerical physics, cyber defense & control implementations
│   └── tests/                 # Full pytest harness (100% empirical pass rate)
│
├── domain_laboratories/       # Specialized multi-disciplinary testbenches
│   ├── aerospace_gnc/         # 15-state ES-EKF, ULA MVDR beamforming, Space Shuttle TMR FDIR, Z3 proofs
│   ├── cyber_forensic_crypto/ # Hardened RISC-V gate model, tamper-evident Merkle blackbox, BFT consensus
│   ├── frugal_mechanics/      # Method of Characteristics (MOC) water-hammer, Seebeck MPPT, 2-RC ECM BMS
│   ├── advanced_quantum_scada/# Tokamak MHD equilibrium, QKD satellite links, Modbus/DNP3 DPI firewalls
│   ├── isomorphic_hybrid/     # Bidirectional state-space physics bridge, silicon DPA/CPA side-channel guard
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

## 🔬 Complete Frontier Hybrid Engines Catalog (Categorized by Domain)

### 1. 🛡️ Post-Quantum Cryptography, ICS/SCADA & Cyber Defense
| Engine | Title | Core Physical / Mathematical Mechanism |
| :--- | :--- | :--- |
| **Engine 04** | PQ-ZK SCADA Self-Healing Sentry | NIST ML-KEM Ring-LWE lattice key encapsulation with Schnorr-Pedersen NIZK telemetry proofs for autonomous ICS PLC quarantine. |
| **Engine 09** | Topological Braid Crypto Ledger | Non-abelian Artin braid group word reduction solving the Conjugacy Search Problem (CSP) for quantum-resistant consensus. |
| **Engine 10** | Gravitational Wave Steganographic Sonar | Matched-filter chirp extraction for covert, low-SNR subsea acoustic steganography. |
| **Engine 66** | Adversarial Active Inference SCADA Sentinel | Variational Free Energy anomaly gating detecting stealth False Data Injection Attacks (FDIA) with autonomous state rollback. |

### 2. 🚀 Aerospace, GNC & Electronic Warfare (EW)
| Engine | Title | Core Physical / Mathematical Mechanism |
| :--- | :--- | :--- |
| **Engine 01** | Clonal Selection UAV Swarm | Affinity maturation clonal selection for Byzantine GPS-denied rendezvous. |
| **Engine 06** | Hypersonic Plasma Sheath RL | Reinforcement learning aerodynamic deflection compensating for blackout ionization sheath. |
| **Engine 14** | Geodesic Solar Sail Navigator | Photon-pressure geodesic transfer orbit optimizer across non-uniform solar radiation fields. |
| **Engine 17** | Memristive HD Computing SAR | Hyperdimensional vector binding for real-time Synthetic Aperture Radar speckle denoising. |
| **Engine 32** | Biomimetic Echolocation LiDAR Fusion | Micro-Doppler bat echolocation acoustic chirps fused with LiDAR pointclouds. |
| **Engine 48** | Micro-Plasma CubeSat Thruster | RF-excited magnetized plasma plume thrust and specific impulse solver. |
| **Engine 51** | Exonuclease DNA LLM Poison Filter | Proofreading 3'-5' exonuclease error correction applied to token stream poisoning filtration. |
| **Engine 52** | Relativistic Squeezed Radar InSAR | Quantum-squeezed entangled microwave interferometry for sub-wavelength topographic mapping. |
| **Engine 65** | Fréchet EVT Anti-Jamming Trajectory | Extreme Value Theory heavy-tailed risk optimizer navigating non-Gaussian EW RF jamming fields. |

### 3. 🧠 Neuromorphic Computing, SNNs & Cognitive AI
| Engine | Title | Core Physical / Mathematical Mechanism |
| :--- | :--- | :--- |
| **Engine 02** | Neuromorphic Jump-Diffusion | Leaky Integrate-and-Fire SNN modeling Poisson jump-diffusion in high-frequency state estimation. |
| **Engine 11** | Epigenetic Neuromorphic Robotics | Dynamic DNA methylation weight masking for continual motor learning without catastrophic forgetting. |
| **Engine 22** | Olfactory Glomerular Gas Tracker | Insect antennal lobe glomerular lateral inhibition SNN for turbulent chemical plume tracking. |
| **Engine 25** | Synaptic Pruning AST Compiler | Developmental synaptic pruning applied to abstract syntax tree Dead Code Elimination (DCE). |
| **Engine 34** | CNT Synaptic Crossbar Accelerator | Carbon nanotube memristive crossbar solving vector-matrix multiplications in-memory. |
| **Engine 43** | Spintronic MRAM SNN Crossbar | Spin-Transfer Torque (STT-MRAM) stochastic switching dynamics for spiking neural layers. |
| **Engine 49** | Neuromorphic Sound Localizer | Interaural Time Difference (ITD) Jeffress coincidence detector with axonal delay lines. |
| **Engine 59** | Active Inference Pain Interceptor | Somatosensory nociceptive predictive coding minimizing variational free energy to preempt phantom limb pain. |
| **Engine 60** | Astrocytic Tripartite SNN | Astrocyte intracellular $Ca^{2+}$ glial feedback modulating spike-timing-dependent plasticity (STDP). |

### 4. ⚛️ Quantum Mechanics, Photonics & Metamaterials
| Engine | Title | Core Physical / Mathematical Mechanism |
| :--- | :--- | :--- |
| **Engine 03** | Quantum Annealing Haptics | Transverse-field Ising Hamiltonian mapped to real-time bilateral teleoperation impedance matching. |
| **Engine 05** | Acoustic Metamaterial Organ-Chip | Phononic crystal acoustic bandgap resonator generating acoustic tweezers in microfluidic channels. |
| **Engine 07** | Photonic DNA Origami Router | DNA-scaffolded dye resonance energy transfer (FRET) for all-optical logic routing. |
| **Engine 16** | Superconducting Flux Qubit Controller | Josephson junction Hamiltonian state control on low-latency bare-metal FPGA registers. |
| **Engine 19** | Quantum Dot Perovskite Harvester | Förster resonance energy transfer (FRET) exciton dynamics in tandem perovskite photovoltaic stacks. |
| **Engine 21** | Metamaterial Cloak RF Harvester | Coordinate-transformation electromagnetic cloak integrating simultaneous RF rectification diodes. |
| **Engine 29** | Dual-Beam Optical Tweezer Sorter | Optical gradient force and dielectrophoretic (DEP) sorting of single micro-particles. |
| **Engine 33** | Quantum Absorption Refrigerator | Lindblad master equation solver for three-level quantum thermal refrigeration cycles. |
| **Engine 41** | Quantum Cascade Laser Spectrometer | Mid-IR cavity ring-down absorption spectroscopy for trace gas isotopic ratio extraction. |
| **Engine 44** | Holographic Photopolymer Storage | Volume holographic storage using coupled-wave diffraction on volume Bragg gratings. |
| **Engine 57** | Quantum-Squeezed DAS Geothermal | Squeezed-state optical distributed acoustic sensing for subterranean micro-fracture triangulation. |
| **Engine 61** | Superconducting Fluxon Debris Sentry| Fluxon vortex pinning dynamics in type-II superconductors for kinetic momentum absorption. |

### 5. 🧬 Synthetic Biology, Extremophiles & Biomimetics
| Engine | Title | Core Physical / Mathematical Mechanism |
| :--- | :--- | :--- |
| **Engine 08** | Physarum Mycelial Smart Grid | Slime mold tubule resistance-conductance network optimization for fault-tolerant grid routing. |
| **Engine 12** | CRISPR Causal Fault Localization | Synthetic guide-RNA cleavage kinetics mapped to causal DAG fault localization in microservices. |
| **Engine 15** | Bioelectric Voltage Morphogenesis | Transepithelial resting potential gradients driving cellular regeneration and pattern formation. |
| **Engine 20** | Chemosynthetic Hydrothermal ALU | Chemolithoautotrophic redox metabolic reaction networks executing binary arithmetic logic. |
| **Engine 23** | Bacterial Quorum Sensing Scheduler | Autoinducer-2 concentration feedback loops for distributed traffic intersection arbitration. |
| **Engine 45** | Extremophile RecA DNA Repair | *Deinococcus radiodurans* RecA double-strand break repair kinetics under intense gamma radiation. |
| **Engine 50** | Morphogenetic Swarm Crystallizer | Turing reaction-diffusion morphogen morphometry coordinating autonomous robot assembly. |
| **Engine 53** | Replicator Phage Bioremediator | Stochastic predator-prey phage-bacteria lysis kinetics in microfluidic pollutant scrubbing. |
| **Engine 58** | Synthetic Myoglobin UUV Oxygen | Reversible oxygen-binding globin kinetics powering non-combustive subsea fuel cells. |
| **Engine 62** | Slime Mold Urban Evacuation Sentry | Adaptive tube-diameter streaming dynamics for congestion-free emergency structural egress. |

### 6. ⚡ Extreme Physics, Fluid Dynamics & Energy Systems
| Engine | Title | Core Physical / Mathematical Mechanism |
| :--- | :--- | :--- |
| **Engine 13** | Thermoacoustic Stirling Cryocooler | Rott's thermoacoustic wave equations for acoustic enthalpy flux refrigeration. |
| **Engine 18** | Relativistic PIC Plasma Wakefield | 1D particle-in-cell laser ponderomotive envelope solver for multi-GeV plasma wakefield acceleration. |
| **Engine 24** | Ferrofluidic Bingham Damper | Non-Newtonian magnetorheological fluid yield stress modulation under seismic shear excitation. |
| **Engine 26** | Magnetohydrodynamic Molten Salt Pump| Coupled Navier-Stokes and Maxwell-Ampere equations for conductive fluid Lorentz pumping. |
| **Engine 27** | Liquid Crystal Elastomer Actuator | Nematogenic order parameter Frank elasticity driving opto-thermally actuated soft robotics. |
| **Engine 28** | Turbomachinery Piezo Harvester | Coupled Euler-Bernoulli beam piezo-elastic resonance harvesting blade flutter energy. |
| **Engine 30** | Stochastic Resonance Amplifier | Bistable Duffing potential Kramers rate switching for sub-threshold signal detection. |
| **Engine 31** | Plasma Electrolytic Nanocoater | Micro-discharge dielectric breakdown dynamics producing ceramic nanocoatings on light alloys. |
| **Engine 35** | 3D CA Premixed Flame Combustor | Cellular automata thermal-diffusive Kuramoto-Sivashinsky flame front propagation. |
| **Engine 37** | Sonoluminescence Cavitation Reactor| Keller-Miksis bubble dynamics modeling acoustic cavitation picosecond plasma flashes. |
| **Engine 38** | Magnetostrictive Terfenol-D Sonar | Non-linear Jiles-Atherton ferromagnetic hysteresis for high-power underwater transducers. |
| **Engine 40** | MOF-801 Atmospheric Water Harvester| Langmuir adsorption isotherm and heat transfer kinetics for low-humidity water harvesting. |
| **Engine 46** | Radioisotope Seebeck Space RTG | Transient thermal diffusion and Thomson effect in multijunction thermoelectric generators. |
| **Engine 47** | Nanofluidic Memristive Osmotic Cell | Debye layer overlapping and streaming potential electrokinetics in 2D graphene nanochannels. |
| **Engine 54** | Superconducting $sCO_2$ Datacenter Loop | Closed-loop Brayton supercritical $CO_2$ thermodynamic cycle cooling dense compute clusters. |
| **Engine 55** | Agentic Control Barrier Function (CBF)| Quadratic programming CBF safety filter preventing flash-crash cascading liquidation loops. |
| **Engine 56** | Circadian Endocrine Microgrid | Melatonin-cortisol diurnal hormonal rhythms modulating renewable energy storage dispatch. |
| **Engine 63** | Stomatal Turgor Architectural HVAC | Plant guard-cell osmotic turgor pressure mechanics for zero-power thermal building ventilation. |

### 7. 🫀 Biomedical Physics & Diagnostics
| Engine | Title | Core Physical / Mathematical Mechanism |
| :--- | :--- | :--- |
| **Engine 36** | Optogenetic Cardiac Defibrillator | FitzHugh-Nagumo cardiac action potential suppression using light-gated Channelrhodopsin-2. |
| **Engine 39** | Neuromuscular Prosthetic Driver | Hill-type three-element muscle tendon model with recursive closed-loop myoelectric feedback. |
| **Engine 42** | Piezoresistive Electronic Skin | Percolation threshold conductive elastomer piezoresistive array for tactile slip detection. |
| **Engine 64** | Non-Hermitian EP Sepsis Detector | Exceptional point (EP) second-order eigenvalue splitting for sub-picomolar systemic sepsis detection. |

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

All 66 frontier hybrid engines, domain laboratories, and continuum testbenches execute locally and verify zero runtime errors and numerical convergence.

---

## 📜 Author

**Ali Malik** ([@am-LLM](https://github.com/am-LLM))  
*Cross-domain systems architecture, low-resource hardware/software engineering, and physics-informed computational modeling.*
