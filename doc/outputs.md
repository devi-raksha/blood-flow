# Outputs

Output is controlled by `Output filename` and `Output directory` in the `MetricFlowSystem<1, 3>` section. An empty output directory writes into the process working directory; otherwise the implementation prefixes generated names with that directory.

## Visualization files

At each IDA output callback, deal.II `DataOut` writes parallel visualization records:

- one `.vtu` piece per MPI rank;
- a `.pvtu` record describing the pieces;
- a `.pvd` time-series file using the configured output base name.

The visualization vectors include the FE-range solution and computed pressure fields. The exact piece suffixes are produced by deal.II and include the output cycle and MPI rank.

## Vessel CSV probes

At setup, the implementation opens one CSV file per vessel at its arc-length midpoint. The current filename pattern is:

```text
HDG_IDA_Vessel_<vessel-id>.csv
```

Rows are appended at IDA output times. The header and columns are emitted by `BloodFlowSystem::open_csv_files()`/`BloodFlowSystem::write_csv_row()` in `source/blood_flow_system.cc`; consumers should treat them as implementation output rather than a stable file-format contract.

## Logs and parameter receipt

`ParameterAcceptor::initialize()` writes `last_used_parameters.prm` after reading the input. deal.II logging is controlled by `Verbosity (console depth)`, and rank zero is the visible writer in the MPI run.

## Status

The repository contains example images and reference assets. This page documents file-writing behavior.
