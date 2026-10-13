# ADAN56 benchmark

## 1. Purpose

This directory packages the repository's ADAN56 arterial-network input as a
reproducible tutorial and topology demonstration. The maintainer clarification
identifies this case as **56 anatomical arteries represented by 77 computational
vessel segments**. It does not assert that the solver has completed a benchmark
run or that a numerical result has been validated.

## 2. Scope

The tutorial covers the source-backed ASCII VTK network, the existing 56_ADNR
parameter source, input validation, topology inspection, and a deterministic
connectivity diagram. It does not change solver behavior, funding, licensing,
or mathematical model decisions.

## 3. Non-goals

This is not a runtime acceptance report, convergence study, or quantitative
validation claim. No benchmark curves, thresholds, literature comparison, or raw
reference data are invented here. Image-only results associated with the tracked
56-artery assets are demonstration/reference material unless separate provenance
establishes more.

## 4. Prerequisites

A POSIX shell and Python 3 are required for input validation and diagram
regeneration. A configured and built `metric_flow_x` executable with deal.II,
SUNDIALS/IDA, and its selected linear-algebra backend is additionally required
for a solver run. The local environment may not provide that executable.

## 5. Input assets

`network.vtk` is a byte-for-byte copy of `notebooks/56_adnr_new.vtk`, the
provenance-supported 56_ADNR variant selected by the existing parameter source.
`parameters.prm.in` is derived from
`parameters/benchmark-parameters/56_ADNR/56_adnr.prm`; its mesh path is made
portable with `@SOURCE_DIR@/tutorials/04_adan56/network.vtk` and its output
directory is dedicated to this tutorial. The template is retained as `.prm.in`;
`run.sh` expands `@SOURCE_DIR@` into a temporary parameter file.

The selected VTK has the exact cell arrays `vessel_id`, `a0`, `a_d`, `E`,
`h_wall`, `p_d`, `p0`, `L`, and `r_d`, and the exact point arrays `boundary_id`,
`R1`, `R2`, `C`, and `P_out`. The alternative `notebooks/56_adnr.vtk` has the
same points and connectivity but also contains `r_in` and `r_out`; it was not
silently substituted for the selected no-taper variant.

## 6. Topology

The validator reports 78 points, 77 `VTK_LINE` cells, and vessel IDs 0 through
76. The boundary array contains inlet ID 0, junction ID 255, and terminal IDs
1 through 31. Thus **56 anatomical arteries** are represented computationally
by **77 vessel segments** in this network. `topology.json` and `network.svg`
are generated from the VTK connectivity; they are not hand-authored topology.

## 7. Equations (77 computational vessel segments)

The source assembles the network junction equations for each junction from the
active graph. This tutorial records the source contract for the 77-segment
network; it does not introduce an alternate equation set or claim mathematical
verification. Equation and orientation details remain those implemented by
`assemble_trace_junction_equations` in `source/blood_flow_system.cc`.

## 8. Parameter configuration

CMake expands `@SOURCE_DIR@` in parameter templates. The local template keeps
that token so it remains portable across checkouts. `run.sh` performs the
equivalent expansion into a temporary file and runs from the repository root.
The parameter source retains its configured 1.0-time-unit inflow expression and
ADNR constants. These are input metadata, not validation thresholds or claims
about the provenance of raw supporting data.

## 9. Run

Validate and attempt the tutorial from the repository root or this directory:

```bash
./tutorials/04_adan56/run.sh
# or select an executable explicitly:
METRIC_FLOW_X_EXECUTABLE=/path/to/metric_flow_x ./tutorials/04_adan56/run.sh
```

The script validates the copied VTK and parameter path before invoking the
solver. If no configured executable exists, it exits nonzero with an actionable
error and explicitly states that no solver run occurred; it never claims runtime
success. A solver invocation, when available, remains an unvalidated
reproducibility demonstration unless raw outputs and approved comparison data
are supplied.

## 10. Generated diagram

Regenerate the deterministic SVG and topology JSON with:

```bash
python3 tools/generate_network_diagram.py \
  tutorials/04_adan56/network.vtk \
  --svg tutorials/04_adan56/network.svg \
  --json tutorials/04_adan56/topology.json \
  --source notebooks/56_adnr_new.vtk \
  --schema blood-flow.adan56.topology.v1
```

The command validates first and derives labels from VTK connectivity and
boundary arrays. It does not draw a hand-authored illustration.

## 11. Diagnostics and postprocessing

This tutorial provides input and topology diagnostics only. No raw pressure,
flow, or reference-curve table is checked in, and no postprocessing command is
presented as a benchmark acceptance test. Existing image-only artifacts under
`NumData/56-arteries/` and related benchmark-parameter directories may be used
as demonstration/reference material only until their input, run, and raw-data
provenance is established.

## 12. Validation

Validate the copied input independently with:

```bash
python3 tools/validate_vtk_network.py tutorials/04_adan56/network.vtk
```

The checked-in topology JSON and SVG were generated only after this validator
returned `valid: true`. Focused checks also compare the copied VTK byte-for-byte
with its selected source and inspect the expected arrays and connectivity.
These checks establish input integrity, not solver runtime acceptance.

## 13. Provenance

`reference/provenance.yml` records the selected source path, copied-asset SHA-256,
parameter-source path and digest, exact arrays/topology facts, validator, and
known variant distinction. The repository does not provide raw reference data,
a complete literature citation, or a run receipt for this tutorial. Those
provenance items remain visibly pending. Image-only 56 results are recorded as
demonstration/reference pointers, not quantitative validation evidence.

## 14. Limitations and status

The proposed message is **WP-11: add ADAN56 benchmark tutorial**. The repository
now has a source-backed ADAN56 input and inspectable 77-segment topology for the
clarified identity of 56 anatomical arteries. It does not claim a completed
runtime, validation curves, acceptance threshold, raw reference data, or
literature-backed benchmark comparison. Missing source/license details and raw
reference/run provenance remain maintainer follow-ups; this work unit does not
alter those decisions.
