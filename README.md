# PINNs for Mechanical Engineers

**Physics-Informed Neural Networks, applied to the problems mechanical engineers actually deal with — fluids, heat, solids, and the messy places where they couple.**

*Python · PyTorch · practitioner-grade, runnable, and honest about where PINNs strain.*

---

## Overview

This is a hands-on tutorial series that builds physics-informed neural networks (PINNs) from working code, one mechanical-engineering problem at a time. Every tutorial is a self-contained Jupyter notebook that:

- solves a **real** governing equation (Navier–Stokes, heat conduction, elasticity, Boussinesq convection, beam bending) — not a toy,
- **validates** the result against a reference — an exact analytical solution where one exists, a published benchmark otherwise,
- ships with **real training logs and metrics** — no hand-waving, no fabricated numbers,
- and stays **honest about the limits**: each notebook says plainly where a classical solver still beats a PINN, and where PINNs genuinely earn their keep.

The series assumes you're comfortable with the PINN fundamentals (autograd, collocation points, the residual loss, the Adam → L-BFGS recipe). If you're starting from zero, see [**The Foundational Series**](#the-foundational-series) below.

---

## The Series

| # | Tutorial | Focus | Reference | Status |
|---|----------|-------|-----------|--------|
| 1 | **2D Navier–Stokes — Lid-Driven Cavity** | coupled, nonlinear flow; pressure gauge | Ghia, Ghia & Shin (1982) | ✅ |
| 2 | **2D Steady Heat Conduction** | mixed Dirichlet/Neumann BCs | analytical (separation of variables) | ✅ |
| 3 | **Linear Elasticity — Plate with a Hole** | vector fields; stress via autograd; stress concentration | Kirsch (1898) | ✅ |
| 4 | **Natural Convection — Buoyant Cavity** | multiphysics coupling; stream-function/vorticity; hard constraints | de Vahl Davis (1983); Fowler, McDevitt & Roy (2024) | ✅ |
| 5 | **Euler–Bernoulli Beam** | 4th-order derivatives; hard-constrained BCs | analytical (Euler–Bernoulli) | ✅ |
| 6 | **Backward-Facing Step** | flow separation & reattachment | Armaly et al. (1983) | ✅ |
| 7 | **Unsteady Vortex Shedding — Cylinder Wake** | unsteady flow; data assimilation; the failure modes | Schäfer & Turek (1996); Raissi et al. (2019) | ✅ |

Each tutorial is standalone — you can start anywhere — but the difficulty and the ideas build across the series.

---

## What's Inside

**Tutorial 1 — Lid-Driven Cavity.** The series opens at the deep end: the steady incompressible Navier–Stokes equations with three coupled outputs (u, v, p), three residuals, the pressure-gauge problem, and the cavity corner singularity. Validated against the canonical Ghia et al. centerline data.

**Tutorial 2 — Heat Conduction.** A deliberate step down to a single linear scalar equation (Laplace), chosen because it has an *exact* answer. Introduces mixed boundary conditions — Dirichlet walls plus an insulated (Neumann) edge — and validates to sub-0.1% against a closed-form field.

**Tutorial 3 — Elasticity, Plate with a Hole.** First vector-valued output (displacements u, v), with stress recovered by autograd. A curved traction-free boundary and quarter-symmetry, validated against the Kirsch solution — including the famous stress-concentration factor of **3** at the hole edge.

**Tutorial 4 — Natural Convection.** The multiphysics tutorial: buoyancy-driven flow where temperature and velocity feed back on each other. Uses the **stream-function/vorticity** formulation with **hard boundary constraints** (following Fowler et al. 2024) to escape the pure-conduction trap that defeats the naive primitive-variable approach. Validated against the de Vahl Davis Nusselt-number benchmark.

**Tutorial 5 — Euler–Bernoulli Beam.** One equation, one field, one dimension — but *fourth order*. The cleanest laboratory in the series for high-order-derivative behavior: four nested autograd calls, four boundary conditions in two physical pairs, a hard-constrained fixed end, and an honest look at the accuracy ceiling that high derivatives impose.

**Tutorial 6 — Backward-Facing Step.** Back to fluids, and the last "well-behaved" one: steady, laminar internal flow over a sudden expansion — the classic geometry behind diffusers, sudden pipe enlargements, combustor dumps, and HVAC ducts. As the flow passes the step it **separates** from the wall and **reattaches** downstream, enclosing a recirculation bubble the network has to discover rather than be told about. The validation target is a single clean scalar — the **reattachment length** as a function of Reynolds number — checked against the Armaly et al. (1983) experiments, where it grows roughly linearly in the laminar regime.

**Tutorial 7 — Unsteady Vortex Shedding.** The capstone, and the hardest problem in the series. Flow past a circular cylinder at Re ≈ 100 sheds the alternating von Kármán vortex street — an *unsteady*, time-dependent flow (time enters as an extra input coordinate). This is where the honesty of the series pays off: a **pure forward PINN fails** here, converging to the unstable *symmetric* steady wake (which also satisfies the equations) instead of developing the oscillating shedding. The fix — and the tutorial's real subject — is **data assimilation**: feeding the network a handful of observations to break the symmetry, exactly as in the celebrated Raissi et al. (2019) cylinder result. Validated against the Schäfer & Turek (1996) benchmark (drag/lift coefficients, Strouhal number St ≈ 0.2). The finale delivers the series' central thesis: forward PINNs often lose to classical CFD, but *this* — assimilating sparse data into full-field, differentiable physics — is where they genuinely earn their keep.

---

## Getting Started

### Requirements

Minimal by design — three libraries:

```
torch
numpy
matplotlib
```

Python 3.9+ recommended. A CUDA GPU is optional for most tutorials and **recommended for Tutorial 4** (natural convection is a heavy, 4th-order-autograd solve).

### Installation

```bash
git clone https://github.com/rraj-io/PINN_for_Mechanical_Engineer.git
cd PINN_for_Mechanical_Engineer
pip install torch numpy matplotlib jupyter
```

### Running

```bash
jupyter notebook
```

Open any tutorial notebook and run it top to bottom. Each notebook automatically uses a GPU if one is available, and falls back to CPU otherwise. Training logs, validation metrics, and figures are produced inline.

---

## Repository Structure

```
PINN_for_Mechanical_Engineer/
├── PME_Tutorial_01/PME_Tutorial_01_lid_driven_cavity.ipynb/
├── PME_Tutorial_02/PME_Tutorial_02_heat_conduction.ipynb/
├── PME_Tutorial_03/PME_Tutorial_03_linear_elasticity.ipynb/
├── PME_Tutorial_04/PME_Tutorial_04_natural_convection.ipynb/
├── PME_Tutorial_05/PME_Tutorial_05_euler_bernoulli_beam.ipynb/
├── PME_Tutorial_06/PME_Tutorial_06_backward_facing_step.ipynb
├── PME_Tutorial_07/PME_Tutorial_07_vortex_shedding.ipynb
└── README.md
```

Each tutorial folder contains the runnable notebook and any figures/assets it generates.

---

## Philosophy: honest about where PINNs win — and don't

This series takes a deliberate stance: **a PINN is not always the right tool, and the tutorials say so.** For clean, forward, well-posed problems (steady heat conduction, a cantilever beam), a classical finite-difference or finite-element solver is faster and more accurate, and each notebook states this plainly. Where PINNs genuinely earn their keep is elsewhere — mesh-free handling of awkward geometry, derived quantities and parameter sensitivities for free via autograd, and problems where the physics is only partially known. Being candid about the trade-off is the point, not a caveat.

---

## The Foundational Series

New to PINNs, or want the build-it-from-scratch treatment in one place? **[*PINNs from Scratch*](https://rajrohit403.gumroad.com/l/pinn_tutorial)** is a 100+ page ebook covering the fundamentals this series builds on — from your first `autograd` call through 1D ODEs, the Poisson equation, and nonlinear Burgers, with runnable notebooks throughout.

---

## References

- Ghia, U., Ghia, K. N. & Shin, C. T. (1982). *High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method.* J. Comput. Phys. **48**, 387–411.
- Kirsch, G. (1898). *Die Theorie der Elastizität und die Bedürfnisse der Festigkeitslehre.* VDI Z. **42**, 797–807. (See also Timoshenko & Goodier, *Theory of Elasticity*.)
- de Vahl Davis, G. (1983). *Natural convection of air in a square cavity: a bench mark numerical solution.* Int. J. Numer. Methods Fluids **3**, 249–264.
- Fowler, E., McDevitt, C. J. & Roy, S. (2024). *Physics-informed neural network simulation of thermal cavity flow.* Sci. Rep. **14**, 15203.
- Armaly, B. F., Durst, F., Pereira, J. C. F. & Schönung, B. (1983). *Experimental and theoretical investigation of backward-facing step flow.* J. Fluid Mech. **127**, 473–496.
- Schäfer, M. & Turek, S. (1996). *Benchmark computations of laminar flow around a cylinder.* Notes Numer. Fluid Mech. **52**, 547–566.
- Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). *Physics-informed neural networks.* J. Comput. Phys. **378**, 686–707.

---

## Links

- 📓 **Substack** (articles & notes on PINNs, neural operators, CFD, SciML): [rajrohit403.substack.com](https://rajrohit403.substack.com/)
- 📘 **Ebook** — *PINNs from Scratch*: [rajrohit403.gumroad.com/l/pinn_tutorial](https://rajrohit403.gumroad.com/l/pinn_tutorial)
- 💻 **This repo**: [github.com/rraj-io/PINN_for_Mechanical_Engineer](https://github.com/rraj-io/PINN_for_Mechanical_Engineer)

---

## Citation

If this series is useful in your work or teaching, a link back is appreciated:

```
Rohit Raj, "PINNs for Mechanical Engineers," HalfQubiX, 2026.
https://github.com/rraj-io/PINN_for_Mechanical_Engineer
```

## License

Released under the MIT License — see `LICENSE`. You're free to use, modify, and build on the code; attribution is appreciated.

---

*Built by Rohit Raj · HalfQubiX — translating CFD, FEA, and scientific machine learning into runnable, practitioner-grade tutorials.*
