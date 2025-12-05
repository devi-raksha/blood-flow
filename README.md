# Blood Flow DG/ARKode Solver

This project solves a 1D blood–flow model embedded in 3D using discontinuous Galerkin (DG) spatial discretization and SUNDIALS ARKode for time integration. The formulation follows the notes in `latex/blood_flow.tex`.

## What We Solve

- Unknowns: cross-sectional area `A` and mean velocity `U` along a 1D centerline embedded in 3D.
- Governing equations:
  - Area: `∂_t A + ∇·(b A U) = 0`
  - Momentum: `∂_t U + ∇·(b (U^2/2 + P(A)/ρ)) + η_c U = 0`
- Tube law (pressure–area relation): `P(A) = p0 + μ[(A/A0)^m – 1]`, wave speed `c = sqrt(A/ρ dP/dA)`.

## Numerical Method

- **Spatial discretization:** DG on the 1D mesh embedded in 3D, with Lax–Friedrichs/HLL numerical fluxes and characteristic boundary conditions.
- **Time integration:** ARKode (deal.II wrapper) in fully implicit mode (`f_E = 0`, `f_I = L`). Mass matrix supplied; Jacobian assembled explicitly; UMFPack solves for the Newton systems and mass solves.
- **Linearization:** Newton systems use `N = M – γ J`, with exact Jacobian of the implicit residual; viscosity term included.

## Repository Layout

- `source/`, `include/`: Implementation of `BloodFlowSystem` and utilities.
- `apps/`: One executable per `.cc` driver (CMake auto-generates targets).
- `parameters/`: Sample parameter files, including ARKode settings, output directory, verbosity, and tube-law constants.
- `tests/`: Unit/regression tests (DG residuals, Jacobian finite-difference check).
- `latex/blood_flow.tex`: Detailed mathematical derivation.

## Building

```bash
cmake -S . -B build
cmake --build build
```

Executables are generated for each file in `apps/`, linked against the shared library built from `source/*.cc`.

## Running

Pass a parameter file to an app, e.g.:

```bash
./build/blood_flow params/parameters.prm
```

Key runtime settings (see parameter files):

- `Output filename`, `Output directory` for VTU/PVD output.
- `Verbosity (console depth)` to control `deallog`.
- ARKode block: initial/final time, tolerances, step-size hints, implicit/explicit flags.

## Testing

```bash
ctest --output-on-failure
```

Tests include residual checks, mass integration, and a finite-difference Jacobian comparison (`tests/test_jacobian_fd`).

## Notes on Boundary Conditions

Characteristic-based mapping of exterior states is used for subcritical/supercritical in/outflow. Residual and Jacobian assemble the same flux/viscosity terms; signs follow the formulation in `latex/blood_flow.tex`.

## References

- Formulation and linearization: `latex/blood_flow.tex`.
- ARKode wrapper: deal.II `SUNDIALS::ARKode`.

## License

This project is licensed under the MIT License (see `LICENSE.md`).
