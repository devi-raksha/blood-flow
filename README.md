# MetricFlowX

MetricFlowX is a standalone deal.II library for the one-dimensional blood-flow
DAE embedded in three-dimensional geometry. It contains the distributed mesh,
DG/HDG discretization, RCR terminal models, junction equations, residual
assembly, and analytic linearizations. It has no ImmersX dependency and does
not contain coupling or TensorProduct code.

The mathematical Problem is

```text
F(t, y, y_dot) = 0
```

where the native state contains cell area and velocity, face-trace area and
velocity, and (when present) RCR capacitor pressures. Pressure is the derived
observable `P(A)`; it is not an additional DAE unknown.

## Public API

```cpp
#include <metric_flow_x/blood_flow_system.h>

MetricFlowX::BloodFlowSystem<1, 3> flow(MPI_COMM_WORLD);
flow.initialize_params("network.prm");
flow.setup();

auto y     = flow.make_state();
auto y_dot = flow.make_state();
auto F     = flow.make_state();

flow.initialize_state(y, 0.0);
flow.initialize_state_derivative(y_dot, 0.0);
flow.assemble_residual(0.0, y, y_dot, F);
flow.assemble_state_jacobian(0.0, y, y_dot);      // dF/dy
flow.assemble_derivative_jacobian(0.0, y, y_dot); // dF/dy_dot
```

`triangulation()`, `dof_handler()`, `finite_element()`, `constraints()`,
`mpi_communicator()`, ownership index sets, differential/algebraic masks,
component DoFs, and area/velocity extractors expose the discretization needed
by an external multiphysics application. Constitutive access is provided by
`vessel_properties()`, `pressure()`, `pressure_derivative()`, and
`wave_speed()`.

The state layout is deliberately not split into separate area and velocity
vectors. Component values are identified with
`BloodFlowSystem<1, 3>::Component`, preserving the native distributed layout.

## Standalone application

The `blood_flow` executable reads a parameter file, delegates SUNDIALS IDA
time integration to its standalone `BloodFlowIDARunner`, and writes the
established VTU/PVD and CSV outputs. The library Problem owns physics and
spatial assembly; applications embedding it can own time integration and call
the residual/Jacobian API directly.

## Building

MetricFlowX uses an out-of-source CMake build and requires deal.II 9.5 or newer:

```bash
cmake -S . -B build-metric-flow-x \
  -DDEAL_II_DIR=/path/to/deal.II \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-metric-flow-x
```

The installed package exports the target `MetricFlowX::metric_flow_x`:

```bash
cmake --install build-metric-flow-x --prefix /path/to/install
```

```cmake
find_package(MetricFlowX CONFIG REQUIRED)
target_link_libraries(application PRIVATE MetricFlowX::metric_flow_x)
```

Headers are installed below `include/metric_flow_x/`, libraries below `lib/`,
and the standalone executable below `bin/`.

## Large network datasets

The generated vascular-network datasets are distributed as a GitHub Release asset so they do not inflate the Git checkout. The archive is about 399 MB; its contents occupy about 555 MB when extracted.

From the repository root, run:

```bash
bash scripts/download_datasets.sh
```

The script downloads `blood-flow-datasets-v0.1.0.tar.gz` from release `v0.1.0` and extracts the files into their expected `notebooks/` paths. The downloaded data paths are ignored by Git.

To extract into another directory, pass it as the first argument:

```bash
bash scripts/download_datasets.sh /path/to/dataset-root
```

## Testing

```bash
ctest --test-dir build-metric-flow-x --output-on-failure
```

The suite covers residual and Jacobian finite differences, mass assembly,
junctions, VTK input, and a coupling-readiness test that exercises state
creation, DAE masks, component extraction, split Jacobians, and constitutive
observables without using ImmersX.

The formulation and derivation remain documented in `doc/math.md` and
`latex/blood_flow.tex`.

## License

This project is licensed under the MIT License (see `LICENSE.md`).
