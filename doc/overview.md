# Overview

This project solves a one-dimensional blood-flow model on a network of line segments embedded in three-dimensional space. The current application is `apps/blood_flow.cc`, which instantiates `BloodFlowSystem<1, 3>` and reads a deal.II parameter file before constructing the mesh and running IDA.

## Unknown layout

The implementation uses a hybridized, HDG-type finite-element layout:

- discontinuous cell fields represent area `A` and axial velocity `U`;
- a pair of face trace fields represents `A_hat` and `U_hat` on every unique face;
- RCR terminal capacitor pressures, when present, occupy an appended block.

These blocks are assembled into one distributed vector. IDA treats cell and capacitor rows as differential and trace rows as algebraic.

## Discretization and time integration

The source provides HLL, HLL-HDG, and Lax-Friedrichs flux implementations and their linearizations. Select them with the exact strings `HLL`, `HLL_HDG`, or `LAX_FRIEDRICHS`. The time integrator is SUNDIALS IDA, not ARKode. Newton systems use either the configured direct PETSc/Trilinos path or GMRES with ILU; see [Configuration](configuration.md).

For installation and commands, see [Installation and build/run](installation.md). For generated files, see [Outputs](outputs.md).

```{include} background.md
```

```{include} repository-layout.md
```
