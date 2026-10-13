# WP-10: 37-segment arterial-network demonstration

## 1. Purpose

This directory packages the repository's 37-segment arterial network as a reproducible
input and diagnostics demonstration. It makes the copied VTK, parameter template,
topology record, and periodicity calculation inspectable without presenting a
validated benchmark.

## 2. Scope

The tutorial covers the legacy ASCII VTK network, the current IDA parameter schema,
topology validation, and comparison of two final periods in explicitly structured
CSV traces. It does not change solver behavior, funding, licensing, or mathematical
model decisions.

## 3. Non-goals

This is not a benchmark, convergence study, or validation result. No numerical
comparison to a reference curve is reported: raw reference curves and their
provenance are absent. The name **37-segment arterial network** describes the active
network topology; this tutorial does not claim tapering.

## 4. Prerequisites

A POSIX shell and Python 3 are required for input validation and diagnostics. A
configured and built `metric_flow_x` executable with deal.II, SUNDIALS/IDA, and its
selected linear-algebra backend is additionally required for a solver run. The local
environment may not provide that executable. The periodicity tool uses only the
Python standard library.

## 5. Input assets

`network.vtk` is a byte-for-byte copy of `notebooks/37_vessel_network.vtk`.
`parameters.prm.in` is the canonical 37-segment parameter template with the mesh
path set to `@SOURCE_DIR@/tutorials/03_37_arteries/network.vtk` and a dedicated
output directory. The template is intentionally retained as a `.prm.in`; `run.sh`
expands `@SOURCE_DIR@` into a temporary file for a direct local invocation.

## 6. Topology

The validator reports 38 points, 37 `VTK_LINE` cells, and 37 vessel IDs. Terminal
boundary IDs are 1 through 16 (the inlet is boundary ID 0 and junction points use
ID 255). `topology.json` and `network.svg` are generated from the VTK connectivity;
`reference/provenance.yml` records the source digest and validation facts. The
active VTK contains no `r_in`/`r_out` taper arrays, so this asset is not described as
a tapering network.

## 7. Equations (K = 37)

The source assembles the same network junction equations for each junction. For a
junction with K incident vessels, the implementation assembles exactly `2K` rows:
one conservation row, `K-1` total-head rows, and K compatibility rows. Here K is
computed from the active graph (37 vessel segments overall); the equation and
orientation details remain those implemented by
`assemble_trace_junction_equations` in `source/blood_flow_system.cc`. This section
records the source contract and does not introduce an alternate equation set.

## 8. Parameter configuration

CMake expands `@SOURCE_DIR@` in parameter templates. The local template keeps that
token so it remains portable across checkouts. `run.sh` performs the equivalent
expansion into a temporary parameter file and runs from the repository root. The
canonical period represented by the inflow expression is 0.827 time units; this is
configuration metadata, not a validation threshold.

## 9. Run

Validate and attempt the tutorial from the repository root or this directory:

```bash
./tutorials/03_37_arteries/run.sh
# or select an executable explicitly:
METRIC_FLOW_X_EXECUTABLE=/path/to/metric_flow_x ./tutorials/03_37_arteries/run.sh
```

The script validates the copied VTK and parameter path before invoking the solver.
If no configured executable exists, it exits nonzero with an actionable error and
explicitly states that no solver run occurred; it never claims a successful run.
A solver invocation, when available, is still a demonstration run rather than a
validated benchmark.

## 10. Generated diagram

Regenerate the deterministic SVG and topology JSON with:

```bash
python3 tools/generate_network_diagram.py \
  tutorials/03_37_arteries/network.vtk \
  --svg tutorials/03_37_arteries/network.svg \
  --json tutorials/03_37_arteries/topology.json \
  --source notebooks/37_vessel_network.vtk \
  --schema blood-flow.37-segment-arterial-network.topology.v1
```

The command validates first and derives labels from VTK connectivity and boundary
arrays. It does not draw a hand-authored topology.

## 11. Diagnostics and postprocessing

`tools/analyze_periodicity.py` consumes only explicitly structured CSV traces with a
`time` column and one or more `trace_*` columns. Supply the documented period; the
tool anchors the final timestamp, constructs the last two complete period windows
algorithmically, linearly interpolates their common phase nodes, and reports the
normalized L2 difference `||candidate-reference||_2 / ||reference||_2`:

```bash
python3 tools/analyze_periodicity.py traces.csv --period 0.827
```

The tool fails clearly when the CSV schema is ambiguous, the period is invalid, or
two complete periods are not present. It does not infer a period or fabricate
reference data. The acceptance threshold is **pending-maintainer-set**; no
numerical threshold is invented here.

## 12. Validation

Validate the copied input independently with:

```bash
python3 tools/validate_vtk_network.py tutorials/03_37_arteries/network.vtk
```

The checked-in topology JSON was generated only after this validator returned
`valid: true`. Run focused tests with:

```bash
python3 -m pytest tests/test_periodicity_tools.py
```

## 13. Provenance

`reference/provenance.yml` records the exact source path, copied-asset path, SHA-256
digest, validator, and topology counts. It also records the absence of raw
comparison curves. The PDFs under `NumData/37-arteries` are image-only plot assets;
they cannot provide reproducible numerical samples, grids, uncertainty, or raw
provenance and therefore are not treated as validation data.

## 14. Limitations and status

The proposed message is **WP-10: document reproducible 37-segment demonstration**.
The repository has topology and input evidence, not a validated result. A configured
build is required for runtime execution, and no runtime success is asserted here.
A future maintainer must provide raw, provenance-linked comparison curves and set an
explicit numerical threshold before this demonstration can support quantitative
validation. The active VTK's missing taper arrays remain an explicit limitation.
