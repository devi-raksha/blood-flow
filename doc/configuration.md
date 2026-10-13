# Configuration

The application uses deal.II's `ParameterAcceptor`. `BloodFlowSystem<1, 3>::initialize_params()` reads the file passed on the command line. For compatibility, its registered parameter section remains named `MetricFlowSystem<1, 3>`; it contains an `IDA parameters` subsection and the `Metric Flow Parameters` constants section. Example files under `parameters/` are the best source for complete syntax; files from older experiments can contain legacy sections and should not be assumed interchangeable.

## System parameters

The following entries are registered by `BloodFlowSystem` in `include/metric_flow_x/blood_flow_system.h` and `source/blood_flow_system.cc`:

| Entry | Purpose |
|---|---|
| `Finite element degree` | Polynomial degree for the cell DG field. |
| `Problem constants` | Constants parameter group used by the model. |
| `Output filename` | Base name for generated visualization output. |
| `Output directory` | Directory prefix for generated output; empty means the current working directory. |
| `Output time interval` | Positive finite values set the interval between written outputs; zero writes at every IDA callback. Positive intervals must be representable at the configured time scale. |
| `Use direct solver` | Select the direct PETSc/ MUMPS or Trilinos direct path; `false` selects GMRES+ILU. |
| `Number of refinement cycles` | Number of solve/refinement cycles in `run()`. |
| `Number of global refinement` | Initial global mesh refinement count. |
| `Theta (penalty parameter)` | Interior penalty/stability parameter used by assembly. |
| `Theta Boundary (stability parameter)` | Boundary stability parameter. |
| `Gamma (Total pressure factor)` | Total-pressure factor used in junction coupling. |
| `Verbosity (console depth)` | deal.II log depth; rank zero controls the visible log. |
| `Numerical flux type` | Exactly `HLL`, `HLL_HDG`, or `LAX_FRIEDRICHS`. Other strings are rejected. |
| `Use Riemann Invariants` | Select the configured boundary treatment using Riemann invariants. |
| `Outlet boundary condition type` | Outlet mode, including the `RCR` mode used by the aortic example. |
| `Use junction mesh` | Select the mesh construction mode for network junctions. |
| `Vtk file path for mesh input` | Legacy VTK network mesh read during setup. |

The `Metric Flow Parameters` group currently registers the source names `rho`, `mu`, `xi`, `m`, and `Rt`, displayed in parameter files as `Density (rho)`, `Viscosity coefficient (mu)`, `Profile constant for friction term (xi)`, `Tube law exponent (m)`, and `Reflection coefficient at outflow boundary (Rt)`.

## IDA parameters

`BloodFlowSystem` delegates the nested `IDA parameters` group to deal.II's `SUNDIALS::IDA` parameter support. Current repository examples use:

- `Initial time`, `Final time`, and `Time interval between each output`;
- `Error control/Absolute error tolerance` and `Relative error tolerance`;
- `Running parameters/Initial step size`, `Minimum step size`, `Maximum number of nonlinear iterations`, and `Maximum order of BDF`.

Additional IDA entries may be accepted by the installed deal.II version. Consult the generated `last_used_parameters.prm` written after initialization and the matching deal.II API for the full backend-specific set. Do not copy the older `ARKOde parameters` subsection from historical sample files into a new IDA run without checking it: the current application constructs `SUNDIALS::IDA`.

## Example

The configured aortic template is a useful starting point:

```text
subsection MetricFlowSystem<1, 3>
  set Numerical flux type = HLL
  set Use direct solver = true
  set Vtk file path for mesh input = .../parameters/aortic.vtk
  subsection IDA parameters
    set Initial time = 0
    set Final time = 1.1
  end
end
```

Use the exact spelling and subsection nesting shown in the parameter files. Parameter defaults in C++ and values in examples are not a scientific calibration.
