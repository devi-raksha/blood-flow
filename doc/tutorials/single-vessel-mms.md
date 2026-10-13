# Single-vessel manufactured solution (MMS)

## Problem solved

This tutorial prepares a one-vessel manufactured-solution (MMS) convergence
study for the current `BloodFlowSystem<1, 3>` implementation. The checked-in
assets contain three exact-solution/RHS profiles (`p1`, `p2`, and `p3`), three
polynomial degrees, a one-cell single-vessel VTK input, and an immutable TXT
reference report. The purpose of this page is to make those inputs and their
provenance reproducible.

The reference tables below are parsed from `convergence.txt`.

## Mathematical model and active assumptions

The source-backed generator derives sources from the volume equations assembled
in [`source/blood_flow_system.cc`](../../source/blood_flow_system.cc):

\[
  A_t + (A U)_x = f_A,
  \qquad
  U_t + \left(\frac{U^2}{2} + \frac{p(A)}{\rho}\right)_x
      - \eta\frac{U}{A} = f_U,
\]

with

\[
  \eta = \frac{2(\xi+2)\pi\mu}{\rho},
  \qquad
  p(A) = p_0 + p_d
      + \frac{4\sqrt{\pi}E h_{wall}}{3a_d}
        (\sqrt{A}-\sqrt{a_d}).
\]

This is a description of the equations currently implemented. The SymPy implementation and its
FunctionParser rendering are in
[`tools/generate_mms_expressions.py`](../../tools/generate_mms_expressions.py).
The generated JSON records the expressions and the pressure law without
silently replacing vessel data with another set of constants.

The three exact fields in the parameter templates are:

| Profile | Exact area `A(x,t)` | Exact velocity `U(x,t)` |
|---|---|---|
| `p1` | `0.1*a0*sin(2*PI*x)*cos(2*PI*t/5) + a0` | `-0.02*a0*sin(2*PI*t/5)*cos(2*PI*x)` |
| `p2` | `t*sin(2*PI*x) + 4` | `cos(2*PI*x)/(2*PI*(t*sin(2*PI*x) + 4))` |
| `p3` | `1 - 0.005*sin(9.42*t - 6.28*x)` | `0.05*sin(6.28*x)` |

These expressions are source-backed inputs.
In particular, the symbolic `a0`, `E`, `h_wall`, `a_d`, `p0`, and `p_d` terms
in the generated RHS require a source interface that the current user
`FunctionParser` does not have.

## Network and physical data

[`tutorials/01_single_vessel_mms`](../../tutorials/01_single_vessel_mms)
contains the three parameter templates and the reference artifacts. The mesh
used by the templates is
[`parameters/single_vessel.vtk`](../../parameters/single_vessel.vtk): two
points, one line cell, and one vessel (`vessel_id = 0`). Its source-backed cell
values are `a0 = 3.0605e-4`, `a_d = 4.5239e-4`, `E = 4e5`,
`h_wall = 1.2e-3`, `p_d = 9.46e3`, `p0 = 0`, `L = 0.24137`, and
`r_d = 0.012`. The point data provide boundary IDs and RCR values.

The physical-unit interpretation used by the implementation is SI for the
state and material quantities: axial coordinate and length in metres (m), time
in seconds (s), area in square metres (m²), velocity in metres per second
(m/s), pressure and Young's modulus in pascals (Pa), wall thickness and radius
in metres, density in kg/m³, and viscosity in Pa·s. The VTK file has no unit
metadata, so these are documented assumptions based on the source conversions.
The literal profile constants above must
be interpreted consistently with those assumptions before a dimensional MMS
run is attempted.

## Boundary and initial conditions

Each template sets `Initial time = 0`, `Final time = 0.1`, and the initial
condition corresponding to its exact profile (see the linked `.prm.in` files).
The templates select `RCR` at the outlet and use the boundary/RCR records in
the VTK input. The boundary conditions correspond to the exact fields at both
ends.

The template RHS is intentionally `0.0; 0.0` as a safe parser-backed baseline.
The generated nonzero source terms must not be substituted until vessel
constants can be passed to the RHS parser. The current code's parser constant
map exposes `rho`, `mu`, `xi`, `m`, and `Rt`, while the pressure law also reads
`E`, `h_wall`, `a_d`, `p0`, and `p_d` from VTK data. This mismatch is the
source-backed blocker for a runnable MMS case.

## Parameter file

The three templates differ in polynomial degree only:

- [`p1.prm.in`](../../tutorials/01_single_vessel_mms/p1.prm.in), degree 1;
- [`p2.prm.in`](../../tutorials/01_single_vessel_mms/p2.prm.in), degree 2;
- [`p3.prm.in`](../../tutorials/01_single_vessel_mms/p3.prm.in), degree 3.

They set `Number of global refinement = 2` and `Number of refinement cycles =
5`. This is the separate **four-cell/five-cycle mesh plan**: starting from the
one-cell VTK and two global refinements, the intended cell counts are 4, 8,
16, 32, and 64.

That cell plan must not be confused with the `DoFs` column in the reference
report. `DoFs` are total degrees of freedom reported by the reference study;
they are not cell counts and are not to be replaced by the four-cell sequence.
The immutable report gives DoF sequences of 32, 64, 128, 256, 512 (degree 1),
40, 80, 160, 320, 640 (degree 2), and 48, 96, 192, 384, 768 (degree 3).
No additional mapping between these two sequences is asserted here.

## Configure and build

From a clean checkout, the source-backed preparation commands are:

```bash
cmake -S . -B build
cmake --build build
```

CMake configures the repository's normal parameter templates under
`build/parameters/`. A runnable preparation can expand `@SOURCE_DIR@` into a
working copy, for example:

```bash
sed "s|@SOURCE_DIR@|$(pwd)|g" \
  tutorials/01_single_vessel_mms/p1.prm.in > /tmp/single-vessel-p1.prm
```

This command prepares a parameter file.

## Run in serial

The eventual serial invocation has the normal executable shape:

```bash
./build/metric_flow_x /tmp/single-vessel-p1.prm
```

The executable accepts the resulting parameter file as its input.

## Run with MPI

After the same implementation blockers are resolved, the corresponding MPI
shape is:

```bash
mpirun -np 2 ./build/metric_flow_x /tmp/single-vessel-p1.prm
```

This is recorded for reproducibility preparation only.

## Expected files and units

The current executable's output behavior is described in
[`doc/outputs.md`](../outputs.md). A successful ordinary run is expected to
write visualization files, `last_used_parameters.prm`, and one vessel probe
CSV named `HDG_IDA_Vessel_<vessel-id>.csv`; the exact runtime output has not
been established for these MMS templates.

The probe CSV header is source-backed and uses these units/conversions:

| Column | Unit |
|---|---|
| `time_s` | s |
| `P_dynpcm2` | dyn/cm² |
| `Q_cm3ps` | cm³/s |
| `A_cm2` | cm² |
| `U_cmps` | cm/s |

The implementation converts pressure from Pa to dyn/cm² by multiplying by 10,
area from m² to cm² by multiplying by `1e4`, velocity from m/s to cm/s by
multiplying by `1e2`, and flow `A U` from m³/s to cm³/s by multiplying by
`1e6`.

## Post-processing commands

The immutable report is
[`reference/convergence.txt`](../../tutorials/01_single_vessel_mms/reference/convergence.txt).
The adjacent
[`reference/convergence.json`](../../tutorials/01_single_vessel_mms/reference/convergence.json)
is its canonical TXT-derived JSON and records SHA-256
`7789933cabd07a015fbbd50d8ff14d293f0831a6f436bfd64b308d76e7e1d3a5`.
Generate the canonical JSON and Markdown table with the checked-in collector:

```bash
python3 tools/collect_convergence.py \
  tutorials/01_single_vessel_mms/reference/convergence.txt \
  --json /tmp/convergence.json
python3 tools/collect_convergence.py \
  tutorials/01_single_vessel_mms/reference/convergence.txt \
  --format markdown --output /tmp/convergence.md
```

The collector retains displayed error and rate strings verbatim. The table in
this page is the collector output. Generate
the source-backed SymPy expressions with:

```bash
python3 tools/generate_mms_expressions.py \
  --output /tmp/mms_expressions.json
```

The generator requires SymPy.

## Validation criteria

blockers are addressed, are:

1. all three cases complete without NaN, negative area, unhandled solver
   failure, or missing output;
2. the generated table preserves the exact reference DoF counts;
3. the final two L2 rates are within 0.10 of `p+1`, and the final two H1 rates
   are within 0.10 of `p` for each degree;
4. numerical errors are compared with the supplied reference using a tolerance
   justified by at least two supported build configurations;
5. the generated documentation table and TXT-derived table are byte-for-byte
   consistent after the documented normalization; and
6. all stated units match parser inputs and output conversions.

## Reference results and provenance

The following three tables are generated from the TXT report via the collector
and canonical JSON described above. `DoFs` remains the
report's total-DoF column and is intentionally separate from the four-cell/
five-cycle mesh plan.

### Polynomial degree 3

| Cycle | DoFs | A L2 | A L2 rate | A H1 | A H1 rate | U L2 | U L2 rate | U H1 | U H1 rate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 2.621e-08 | -- | 1.386e-06 | -- | 2.557e-04 | -- | 1.895e-02 | -- |
| 2 | 96 | 1.676e-09 | 3.97 | 1.796e-07 | 2.95 | 1.657e-05 | 3.95 | 2.438e-03 | 2.96 |
| 3 | 192 | 1.054e-10 | 3.99 | 2.274e-08 | 2.98 | 1.050e-06 | 3.98 | 3.075e-04 | 2.99 |
| 4 | 384 | 6.598e-12 | 4.00 | 2.855e-09 | 2.99 | 6.602e-08 | 3.99 | 3.855e-05 | 3.00 |
| 5 | 768 | 4.299e-13 | 3.94 | 3.574e-10 | 3.00 | 4.267e-09 | 3.95 | 4.825e-06 | 3.00 |

### Polynomial degree 2

| Cycle | DoFs | A L2 | A L2 rate | A H1 | A H1 rate | U L2 | U L2 rate | U H1 | U H1 rate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 40 | 2.729e-07 | -- | 8.754e-06 | -- | 2.554e-03 | -- | 1.295e-01 | -- |
| 2 | 80 | 3.502e-08 | 2.96 | 2.289e-06 | 1.94 | 3.328e-04 | 2.94 | 3.343e-02 | 1.95 |
| 3 | 160 | 4.409e-09 | 2.99 | 5.816e-07 | 1.98 | 4.238e-05 | 2.97 | 8.438e-03 | 1.99 |
| 4 | 320 | 5.523e-10 | 3.00 | 1.463e-07 | 1.99 | 5.343e-06 | 2.99 | 2.116e-03 | 2.00 |
| 5 | 640 | 6.908e-11 | 3.00 | 3.668e-08 | 2.00 | 6.707e-07 | 2.99 | 5.297e-04 | 2.00 |

### Polynomial degree 1

| Cycle | DoFs | A L2 | A L2 rate | A H1 | A H1 rate | U L2 | U L2 rate | U H1 | U H1 rate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 2.197e-06 | -- | 3.316e-05 | -- | 1.926e-02 | -- | 5.766e-01 | -- |
| 2 | 64 | 5.714e-07 | 1.94 | 1.758e-05 | 0.92 | 5.021e-03 | 1.94 | 2.982e-01 | 0.95 |
| 3 | 128 | 1.441e-07 | 1.99 | 8.960e-06 | 0.97 | 1.286e-03 | 1.97 | 1.506e-01 | 0.99 |
| 4 | 256 | 3.609e-08 | 2.00 | 4.514e-06 | 0.99 | 3.256e-04 | 1.98 | 7.549e-02 | 1.00 |
| 5 | 512 | 9.030e-09 | 2.00 | 2.264e-06 | 1.00 | 8.193e-05 | 1.99 | 3.778e-02 | 1.00 |

The JSON provenance records the TXT checksum and identifies the report as the
source of every displayed string. The reference numbers are retained from the
source report.

## Troubleshooting

- **The generator exits with “SymPy is required”.** Install SymPy in the
  interpreter used for the command, then rerun the generator.
- **The generated RHS contains `E`, `h_wall`, `a_d`, `p0`, or `p_d`.** These
  values come from VTK, while the current user RHS parser exposes the constants
  listed in the parameter documentation.
- **The executable appears to run as an ordinary case.** Its output uses the
  formats described in [`doc/outputs.md`](../outputs.md).
- **DoFs do not equal 4, 8, 16, 32, 64.** That is expected: those are cell
  counts in the separate mesh plan, whereas the report's `DoFs` column is total
  degrees of freedom.
- **Documentation build.** This page is MyST Markdown and is included in the
  documentation navigation below.

## Literature

For the repository's implementation crosswalk and model caveats, see
[`doc/math.md`](../math.md) and [`doc/background.md`](../background.md). The
repository bibliography is indexed by [`doc/references.md`](../references.md).
This WP-08 page records the source-backed implementation inputs and reference
provenance.
