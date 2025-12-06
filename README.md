# Blood Flow DG/ARKode Solver

This project solves a 1D blood–flow model embedded in 3D using discontinuous Galerkin (DG) spatial discretization and SUNDIALS ARKode for time integration. The formulation follows the notes in `latex/blood_flow.tex`.

## What We Solve

### Unknowns

- Cross-sectional area: $A(x,t)$
- Mean velocity along the centerline: $U(x,t)$

### Governing equations (conservation form)

Mass (area):
$$
\partial_t A + \nabla\cdot\bigl(b\,A\,U\bigr) = 0
$$

Momentum:
$$
\partial_t U + \nabla\cdot \biggl(b\Bigl(\frac{U^2}{2} + \frac{P(A)}{\rho}\Bigr)\biggr) + \eta_c\,U = 0
$$

### Tube law and wave speed

Pressure–area (tube) law:
$$
P(A) = p_0 + \mu\Bigl[\Bigl(\frac{A}{A_0}\Bigr)^m - 1\Bigr],
\qquad
\frac{dP}{dA} = \mu\,m\,\frac{A^{\,m-1}}{A_0^{\,m}} = \mu\,m\,\frac{1}{A_0}\Bigl(\frac{A}{A_0}\Bigr)^{m-1}.
$$
Wave speed:
$$
c = \sqrt{\frac{A}{\rho}\frac{dP}{dA}}
= \sqrt{\frac{\mu\,m}{\rho}\Bigl(\frac{A}{A_0}\Bigr)^{m}}.
$$

(Here $\rho$ is fluid density, $\eta_c$ a friction coefficient, $b$ a geometric weighting, and $p_0,A_0,\mu,m$ tube-law constants.)

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
