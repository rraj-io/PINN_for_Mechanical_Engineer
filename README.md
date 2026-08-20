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

Ten tutorials in two parts. **Part I** builds forward-solving skill across every physics. **Part II** flips the question — and is where PINNs stop competing with classical solvers and start beating them.

### Part I — Forward problems

| # | Tutorial | Focus | Reference |
|---|----------|-------|-----------|
| 1 | **2D Navier–Stokes — Lid-Driven Cavity** | coupled, nonlinear flow; pressure gauge | Ghia, Ghia & Shin (1982) |
| 2 | **2D Steady Heat Conduction** | mixed Dirichlet/Neumann BCs | analytical (separation of variables) |
| 3 | **Linear Elasticity — Plate with a Hole** | vector fields; stress via autograd; stress concentration | Kirsch (1898) |
| 4 | **Natural Convection — Buoyant Cavity** | multiphysics coupling; stream-function/vorticity; hard constraints | de Vahl Davis (1983); Fowler, McDevitt & Roy (2024) |
| 5 | **Euler–Bernoulli Beam** | 4th-order derivatives; hard-constrained BCs | analytical (Euler–Bernoulli) |
| 6 | **Backward-Facing Step** | flow separation & reattachment | Armaly et al. (1983) |
| 7 | **Transient Heat Conduction** | time as an input; causality; time-windowing | analytical (single-mode decay) |

### Part II — Beyond forward solving

| # | Tutorial | Focus | Reference |
|---|----------|-------|-----------|
| 8 | **The Inverse Problem** | recovering unknown physics from sparse noisy sensors | Raissi et al. (2019); manufactured solution |
| 9 | **The Parametric PINN** | one network for a family of solutions; sensitivities | Sun et al. (2020); analytical (convection–diffusion) |
| 10 | **Conjugate Heat Transfer** | two domains, one interface; flux continuity | analytical (thermal resistances in series) |

Each tutorial is standalone — you can start anywhere — but the difficulty and the ideas build across the series.

---

## What's Inside

**Tutorial 1 — Lid-Driven Cavity.** The series opens at the deep end: the steady incompressible Navier–Stokes equations with three coupled outputs (u, v, p), three residuals, the pressure-gauge problem, and the cavity corner singularity. Validated against the canonical Ghia et al. centerline data.

**Tutorial 2 — Heat Conduction.** A deliberate step down to a single linear scalar equation (Laplace), chosen because it has an *exact* answer. Introduces mixed boundary conditions — Dirichlet walls plus an insulated (Neumann) edge — and validates to sub-0.1% against a closed-form field.

**Tutorial 3 — Elasticity, Plate with a Hole.** First vector-valued output (displacements u, v), with stress recovered by autograd. A curved traction-free boundary and quarter-symmetry, validated against the Kirsch solution — including the famous stress-concentration factor of **3** at the hole edge.

**Tutorial 4 — Natural Convection.** The multiphysics tutorial: buoyancy-driven flow where temperature and velocity feed back on each other. Uses the **stream-function/vorticity** formulation with **hard boundary constraints** (following Fowler et al. 2024) to escape the pure-conduction trap that defeats the naive primitive-variable approach. Validated against the de Vahl Davis Nusselt-number benchmark.

**Tutorial 5 — Euler–Bernoulli Beam.** One equation, one field, one dimension — but *fourth order*. The cleanest laboratory in the series for high-order-derivative behavior: four nested autograd calls, four boundary conditions in two physical pairs, a hard-constrained fixed end, and an honest look at the accuracy ceiling that high derivatives impose.

**Tutorial 6 — Backward-Facing Step.** Back to fluids, and the last "well-behaved" one: steady, laminar internal flow over a sudden expansion — the classic geometry behind diffusers, sudden pipe enlargements, combustor dumps, and HVAC ducts. As the flow passes the step it **separates** from the wall and **reattaches** downstream, enclosing a recirculation bubble the network has to discover rather than be told about. The validation target is a single clean scalar — the **reattachment length** as a function of Reynolds number — checked against the Armaly et al. (1983) experiments.

**Tutorial 7 — Transient Heat Conduction.** Time enters the picture, and the tutorial discovers that adding it is the easy part. A PINN optimizes *all times at once* and has **no sense of causality**, so a single network degrades over a long horizon. The fix is **time-windowing**: a sequence of short intervals, each warm-started from the last and inheriting its initial condition — a PINN that marches through time like a classical solver. Validated to the digit against the exact single-mode decay, with an honest closing note on why diffusion-dominated transients work well and convection-dominated ones remain hard.

**Tutorial 8 — The Inverse Problem.** Where the arrow flips. Instead of computing a field from known physics, we're handed twelve noisy thermometer readings and a governing equation with a **missing coefficient**, and asked to recover both. The entire mechanism is one line — `nn.Parameter`, handed to the same optimizer as the weights — where a finite-element code would need an adjoint solver and a hand-derived sensitivity framework. Recovers the unknown conductivity to within ~1%, and reconstructs the full field as a by-product.

**Tutorial 9 — The Parametric PINN.** The parameter becomes a network *input*, so one training run covers an entire range of operating conditions. The result isn't a solver but a **surrogate**: query any Péclet number in milliseconds, including values never sampled during training, and get **parameter sensitivities** (∂u/∂Pe) straight from autograd — something a mesh solver reaches only by finite-differencing repeated runs. Includes the honest trade: generality bought with precision.

**Tutorial 10 — Conjugate Heat Transfer.** The finale, and the thing mechanical engineers simulate more than almost anything else: a solid cooled by a fluid, where neither domain can be solved without the other. **Two networks, one per material**, coupled only through the two interface conditions — temperature continuity and heat-flux continuity. The true solution has a **kink** at the interface that a single smooth network cannot represent, which is exactly why the two-network approach is the right architecture rather than a convenience.

---

## Getting Started

### Requirements

Minimal by design — three libraries:

```
torch
numpy
matplotlib
```

Python 3.9+ recommended. A CUDA GPU is optional for the lighter tutorials (2, 5, 8, 9, 10) and **strongly recommended for the heavy ones**: Tutorial 4 (natural convection, 4th-order autograd), Tutorial 6 (backward-facing step), and Tutorial 7 (time-windowed marching).

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
├── PME_Tutorial_01/PME_Tutorial_01_lid_driven_cavity.ipynb
├── PME_Tutorial_02/PME_Tutorial_02_heat_conduction.ipynb
├── PME_Tutorial_03/PME_Tutorial_03_linear_elasticity.ipynb
├── PME_Tutorial_04/PME_Tutorial_04_natural_convection.ipynb
├── PME_Tutorial_05/PME_Tutorial_05_euler_bernoulli_beam.ipynb
├── PME_Tutorial_06/PME_Tutorial_06_backward_facing_step.ipynb
├── PME_Tutorial_07/PME_Tutorial_07_transient_heat_conduction.ipynb
├── PME_Tutorial_08/PME_Tutorial_08_inverse_problem.ipynb
├── PME_Tutorial_09/PME_Tutorial_09_parametric_pinn.ipynb
├── PME_Tutorial_10/PME_Tutorial_10_conjugate_heat_transfer.ipynb
└── README.md
```

Each tutorial folder contains the runnable notebook and any figures/assets it generates.

---

## Philosophy: honest about where PINNs win — and don't

This series takes a deliberate stance: **a PINN is not always the right tool, and the tutorials say so.**

Across Part I the verdict is consistent and unflattering. For clean, forward, well-posed problems, a mature classical solver usually wins: a five-line finite-difference stencil beats the PINN on the heat plate, a well-resolved mesh beats it on the plate with a hole, and a finite-volume code solves the buoyant cavity faster and more reliably. Worse, some physics the method cannot reach at all — the naive formulation of natural convection converges confidently to a solution with *no flow in it*, and a forward PINN on vortex shedding collapses onto the steady wake and never sheds. Those are structural failures, not tuning problems, and the tutorials document them rather than hiding them.

Part II is where the promise gets paid. Inverse problems, parametric surrogates, and multi-material coupling are questions a mesh solver wasn't built to answer, and there the comparison stops being close.

The takeaway the series argues for: **a PINN is not a better CFD solver.** It's a mesh-free, differentiable representation of a solution, and its advantages are structural rather than numerical. Knowing *which* problems play to that shape is worth more than any amount of tuning.

---

## The Foundational Series

New to PINNs, or want the build-it-from-scratch treatment in one place? **[*PINNs from Scratch*](https://rajrohit403.gumroad.com/l/pinn_tutorial)** is a 100+ page ebook covering the fundamentals this series builds on — from your first `autograd` call through 1D ODEs, the Poisson equation, and nonlinear Burgers, with runnable notebooks throughout.

---

## References

**Benchmarks validated against**

- Ghia, U., Ghia, K. N. & Shin, C. T. (1982). *High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method.* J. Comput. Phys. **48**, 387–411.
- Kirsch, G. (1898). *Die Theorie der Elastizität und die Bedürfnisse der Festigkeitslehre.* VDI Z. **42**, 797–807. (See also Timoshenko & Goodier, *Theory of Elasticity*.)
- de Vahl Davis, G. (1983). *Natural convection of air in a square cavity: a bench mark numerical solution.* Int. J. Numer. Methods Fluids **3**, 249–264.
- Armaly, B. F., Durst, F., Pereira, J. C. F. & Schönung, B. (1983). *Experimental and theoretical investigation of backward-facing step flow.* J. Fluid Mech. **127**, 473–496.
- Erturk, E. (2008). *Numerical solutions of 2-D steady incompressible flow over a backward-facing step.* Computers & Fluids **37**, 633–655.

**PINN methodology**

- Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). *Physics-informed neural networks.* J. Comput. Phys. **378**, 686–707.
- Fowler, E., McDevitt, C. J. & Roy, S. (2024). *Physics-informed neural network simulation of thermal cavity flow.* Sci. Rep. **14**, 15203.
- Sun, L., Gao, H., Pan, S. & Wang, J.-X. (2020). *Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data.* CMAME **361**, 112732.
- Jagtap, A. D., Kharazmi, E. & Karniadakis, G. E. (2020). *Conservative PINNs on discrete domains for conservation laws.* CMAME **365**, 113028.

**Why PINNs fail, and what to do about it**

- Krishnapriyan, A. et al. (2021). *Characterizing possible failure modes in physics-informed neural networks.* NeurIPS.
- Wang, S., Teng, Y. & Perdikaris, P. (2021). *Understanding and mitigating gradient flow pathologies in PINNs.* SIAM J. Sci. Comput. **43**, A3055–A3081.
- Wang, S., Sankaran, S. & Perdikaris, P. (2022). *Respecting causality is all you need for training PINNs.* arXiv:2203.07404.

**Classical texts**

- Carslaw, H. S. & Jaeger, J. C. (1959). *Conduction of Heat in Solids*, 2nd ed. Oxford University Press.
- Timoshenko, S. P. & Goodier, J. N. (1970). *Theory of Elasticity*, 3rd ed. McGraw–Hill.
- Incropera, F. P. & DeWitt, D. P. *Fundamentals of Heat and Mass Transfer.* Wiley.
- Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow.* Hemisphere.

**Related code**

- [`cmcdevitt2/ThermalCavity`](https://github.com/cmcdevitt2/ThermalCavity) — reference implementation accompanying Fowler et al. (2024), used for Tutorial 4.
- [`nkusla/heat-transfer-pinn`](https://github.com/nkusla/heat-transfer-pinn) — transient heat transfer with PINNs, related to Tutorial 7.
- [`lululxvi/deepxde`](https://github.com/lululxvi/deepxde) — a mature PINN library, useful for comparison.
- [`NVIDIA/modulus`](https://github.com/NVIDIA/modulus) — production-scale physics-ML framework.

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