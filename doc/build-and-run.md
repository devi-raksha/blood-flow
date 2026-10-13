# Build and run

This page is retained as a short command reference. From the repository root:

```bash
cmake -S . -B build
cmake --build build
./build/blood_flow build/parameters/aortic.prm
```

CMake generates configured parameter copies under `build/parameters/`. The runtime command requires a successful local build and suitable deal.II linear-algebra support. See [Installation and build/run](installation.md) and [Configuration](configuration.md) for limitations and parameter details.
