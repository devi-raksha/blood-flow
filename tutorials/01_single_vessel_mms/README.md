# Single-vessel manufactured-solution evidence

This directory contains the reproducibility inputs that can be established from
repository evidence. `reference/convergence.txt` is an immutable copy of the
uploaded report; `reference/convergence.json` is produced by
`tools/collect_convergence.py` and retains every displayed error and rate
string. The JSON provenance records the SHA-256 checksum of the copied text.

`p1.prm.in`, `p2.prm.in`, and `p3.prm.in` are parameter templates for the
source-backed single-vessel mesh (`@SOURCE_DIR@/parameters/single_vessel.vtk`)
and the current IDA parameter schema. They are **not claimed runnable MMS
cases**: the current executable's `FunctionParser` constant map exposes only
`rho`, `mu`, `xi`, `m`, and `Rt`, while the vessel pressure law obtains
`E`, `h_wall`, `a_d`, `p0`, and `p_d` from VTK vessel data. Those vessel-specific
quantities cannot currently be referenced by a user RHS function. In addition,
the local checkout has not had a clean build-and-run of these templates.

`tools/generate_mms_expressions.py` derives the mass and momentum sources from
the equations in `source/blood_flow_system.cc` with SymPy. Its output uses
FunctionParser expression syntax and records this constant-map blocker rather
than silently using a different pressure law. The templates retain zero RHS
values as a safe parser-backed baseline until that source interface is made
explicit; they must not be used as claimed MMS runs.

No convergence values are inferred from local runs. The report values are
reference evidence only.
