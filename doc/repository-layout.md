# Repository layout

The top-level directories have the following roles:

| Path | Contents |
|---|---|
| `apps/` | Application entry points. The current target is `blood_flow` from `blood_flow.cc`. |
| `include/` | Public headers under `metric_flow_x/`, including `BloodFlowSystem` and its supporting data structures. |
| `source/` | C++ implementations for assembly, fluxes, IDA callbacks, output, constants, and VTK utilities. |
| `parameters/` | Source parameter files/templates, VTK meshes, notebooks, and reference assets. |
| `build/` | CMake build tree (generated or local); configured parameter copies are under `build/parameters/`. |
| `tests/` | deal.II test sources and expected-output files. |
| `doc/` | Markdown pages, bibliography, Doxygen configuration, Sphinx configuration, and documentation requirements. |
| `scripts/` | Shell helpers, including `build_doc.sh` and `serve_doc.sh`. |
| `latex/` | Repository TeX manuscript sources. This documentation task does not modify them. |
| `NumData/`, `notebooks/` | Additional data and exploratory/reference material; |

CMake lists the library sources and application target explicitly. `tests/CMakeLists.txt` delegates test registration to deal.II's `DEAL_II_PICKUP_TESTS()` macro.
