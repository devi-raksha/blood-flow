# WP-09: Reproducible Y-junction tutorial

## 1. Purpose

This tutorial packages the repository's source-backed three-vessel Y junction as a
small, inspectable example. It is an input and diagnostics tutorial, not a claim
of model validation or publication status.

## 2. Scope

The tutorial covers the legacy ASCII VTK network, the configured IDA parameter
schema, topology visualization, and residual postprocessing. It does not change
solver behavior, funding, licensing, or model decisions.

## 3. Non-goals

No convergence rate, physical accuracy, image-only result, or successful runtime
is claimed here. In particular, this directory does not use obsolete ARKode
parameter files.

## 4. Prerequisites

A POSIX shell, Python 3, and this checkout are required for input validation and
the diagram tools. A configured and built `metric_flow_x` executable with deal.II,
SUNDIALS/IDA, and its selected linear-algebra backend is additionally required
for a solver run. The local environment may not provide that executable.

## 5. Input assets

`network.vtk` is a byte-for-byte copy of tracked
`notebooks/bifurcation_network.vtk`. `parameters.prm.in` is based on the
configured `parameters/benchmark-parameters/aortic_bifurcation/aortic_bifurcation.prm.in`
template and points to the local copied network after `@SOURCE_DIR@` expansion.

## 6. Topology

The validated graph has four points and three `VTK_LINE` cells. Vessel IDs are
0, 1, and 2. Point 0 is inflow boundary ID 0, point 1 is the interior junction
(ID 255), and points 2 and 3 are terminal boundary IDs 1 and 2. The topology and
source checksum are recorded in `topology.json` and `provenance.json`; the
machine-generated view is `network.svg`.

## 7. Equations (K = 3)

At the interior junction the source assembles exactly `2K = 6` algebraic rows.
Let vessel 0 be the first half-face and vessels 1 and 2 the other half-faces;
`s_i` is the source orientation, and hats denote trace values. The six rows are:

1. **Mass:** `s_0 Ahat_0 Uhat_0 + s_1 Ahat_1 Uhat_1 + s_2 Ahat_2 Uhat_2 = 0`.
2. **Total head (vessel 0 vs 1):** `H_0 - H_1 = 0`.
3. **Total head (vessel 0 vs 2):** `H_0 - H_2 = 0`.
4. **Compatibility (vessel 0):** `Uhat_0 + s_0 4(chat_0 - c0_0) - W_0 = 0`.
5. **Compatibility (vessel 1):** `Uhat_1 + s_1 4(chat_1 - c0_1) - W_1 = 0`.
6. **Compatibility (vessel 2):** `Uhat_2 + s_2 4(chat_2 - c0_2) - W_2 = 0`.

Here the implementation defines `H_i = (gamma/2) Uhat_i^2 + p(Ahat_i)/rho`,
`W_i = U_i + s_i 4(c_i - c0_i)`, and evaluates `c` with the source tube
law. This is a transcription of `assemble_trace_junction_equations` in
`source/blood_flow_system.cc`; it is not a replacement equation set.

## 8. Parameter configuration

CMake expands `@SOURCE_DIR@` for templates under `parameters/`. This tutorial
keeps the same token in its local template. `run.sh` performs the equivalent
local expansion into a temporary parameter file, so the checked-in template is
portable and never embeds an obsolete checkout path.

## 9. Run

Validate and run from the repository root or this directory:

```bash
./tutorials/02_y_junction/run.sh
# or select an executable explicitly:
METRIC_FLOW_X_EXECUTABLE=/path/to/metric_flow_x ./tutorials/02_y_junction/run.sh
```

The script validates the copied network and parameter path first. If no
configured executable exists, it exits nonzero with an actionable error and
explicitly reports that no solver run occurred; it never prints a false success.

## 10. Generated diagram

Regenerate the deterministic SVG and topology JSON with:

```bash
python3 tools/generate_network_diagram.py \
  tutorials/02_y_junction/network.vtk \
  --svg tutorials/02_y_junction/network.svg \
  --json tutorials/02_y_junction/topology.json
```

The command reads VTK connectivity and labels each vessel and boundary ID. It
does not draw a hand-authored topology. (Use `tutorials/02_y_junction/topology.json`
for the output path when copying this command.)

## 11. Diagnostics and postprocessing

`tools/analyze_y_junction_residuals.py` consumes CSV output only when the raw
columns are explicitly named. Its K=3 defaults are `mass_residual`,
`head_residual_0`, `head_residual_1`, and
`compatibility_residual_0..2`; custom names can be supplied with the corresponding
`--*-columns` options. It reports scaled maximum absolute and RMS metrics. If a
required solver residual column is absent, it fails with a useful message and
never fabricates a value. Example:

```bash
python3 tools/analyze_y_junction_residuals.py residuals.csv \
  --mass-scale 1e5 --head-scale 1e3 --compatibility-scale 1e2
```

## 12. Validation

Validate the network independently with:

```bash
python3 tools/validate_vtk_network.py tutorials/02_y_junction/network.vtk
```

The checked-in provenance and topology JSON were generated only after this
validator returned `valid: true`. The validator checks connectivity, required
cell/point arrays, boundary IDs, and positive vessel data.

## 13. Provenance

`provenance.json` records the source path, copied-asset path, validator name,
validation summary, and SHA-256 digest
`4536f94764b94aa4f34fdaac83cd01e4a8e1211f727e7bf72a2f978ba4ca3b58`.
`topology.json` repeats the source digest alongside machine-readable points,
cells, vessel IDs, and boundary IDs. These records make accidental mesh drift
reviewable without inferring topology from the SVG.

## 14. Limitations and status

This is the proposed message **WP-09: add reproducible Y-junction tutorial
assets**. A successful local runtime requires a configured build and is not
asserted by this repository change. Residual metrics are likewise unavailable
until the solver exports the explicitly named residual columns; state columns
are not treated as residuals.
