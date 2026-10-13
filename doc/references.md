# References

The canonical bibliography in [`bibliography/references.bib`](../bibliography/references.bib) contains the literature currently associated with the repository's numerical context. It includes work on one-dimensional blood-flow models, discontinuous Galerkin and hybridizable DG methods, approximate Riemann solvers, and blood-flow benchmarks. The metadata policy and unresolved keys are recorded in [`bibliography/README.md`](../bibliography/README.md).

Selected entries can be cited from MyST using the keys in the bibliography, for example `{cite}`cockburn1998runge` or `{cite}`sherwin2003computational`. The complete bibliography is rendered below.

```{bibliography}
:style: plain
:all:
```

Repository-specific implementation references:

- The executable entry point is `apps/blood_flow.cc`.
- The model and IDA callbacks are implemented in `source/blood_flow_system.cc` and declared in `include/metric_flow_x/blood_flow_system.h`.
- The repository's explanatory equations are in [Mathematics](math.md); the TeX manuscript remains under `latex/`.
