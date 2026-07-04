# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`phasic` is a library for **phase-type distribution algorithms represented as graphs**. A phase-type distribution is the time to absorption of a continuous- or discrete-time Markov chain; here the chain's state space is a directed graph of vertices (states) and weighted edges (transition rates), and the library computes PMFs/PDFs/CDFs, moments, sojourn times, and does Bayesian parameter inference over these models. The main application domain is population genetics (coalescent / two-locus / recombination models), so states typically encode lineage properties (see `state_indexing.py` and `tree_toplogy_encoding.md`).

## Build & environment

The project is managed with **pixi** (conda-based). All commands run inside the pixi environment. The build stack is scikit-build-core + CMake + pybind11, compiling C and C++ into the `phasic_pybind` extension module. MPFR/GMP (high-precision arithmetic) are required on Linux/macOS; OpenMP and JAX/XLA-FFI headers are used when present.

```bash
pixi install                 # create/sync the environment
pixi run install-dev         # build C/C++ + (re)install the package  <-- run after EVERY source change
```

**Critical gotcha: the package is installed as a real copy into `.pixi/envs/default/lib/.../site-packages/phasic`, NOT as an editable install.** Editing anything under `src/phasic/` (Python) or `src/c` / `src/cpp` (native) has no effect until you re-run `pixi run install-dev`. Do not assume a Python-only edit is live.

`install-dev` sets `XLA_FFI_INCLUDE_DIR` from the installed JAX so the C++ FFI handlers compile. If JAX/XLA-FFI headers are missing at build time, the module still builds but JAX integration silently falls back to slower Python callbacks (CMake prints a warning). To get the fast path, JAX must be importable when building.

The root `Makefile` is a standalone one-off compile of an experimental file (`jax_graph_method_pmf.cpp`) and is not part of the normal build.

## Tests

```bash
pixi run test                                        # full suite (also converts tutorial notebooks to scripts first)
pixi run pytest tests/pytest/                        # pytest only
pixi run pytest tests/pytest/test_svgd.py -v         # single file
pixi run pytest tests/pytest/ -k "weight_formula"    # by name pattern
```

Tests live in `tests/pytest/` (`test_*.py`). A global `--timeout=600` is set in `pyproject.toml` as a hang-safety net; the slow SVGD accuracy tests legitimately take ~150–300s. Output is captured with `--capture=tee-sys` (printed *and* captured). Native C/C++ tests are in `tests/cpp/`.

## Architecture (layers, bottom to top)

The stack is four layers; a change in the domain logic usually touches Python only, but performance-critical elimination/FFI lives in C/C++.

1. **C core** — `src/c/`, public header `api/c/phasic.h`. Defines the fundamental graph structs (`ptd_graph`, `ptd_vertex`, `ptd_edge`) plus AVL-tree state lookup, hashing (`phasic_hash.c`), logging, and SCC machinery (`scc_synthetic.c`, `scc_compose.c` — the latter is OpenMP-parallel). Note: `phasic_symbolic.c` (symbolic elimination) is **obsolete and disabled** in the build — the trace-based approach replaced it.

2. **C++ layer** — `src/cpp/`, `api/cpp/`. `phasic::Graph` (`phasiccpp.cpp/.h`), SCC graph (`scc_graph.cpp/.h`), and the **parameterized subsystem** in `src/cpp/parameterized/` (`graph_builder*.cpp`, `ffi_handlers.cpp`) which implements the JAX XLA FFI fast path.

3. **pybind11 bindings** — `src/cpp/phasic_pybind.cpp`, compiled to the `phasic_pybind` extension. Exposes the C++ `Graph` as the Python base class `_Graph`.

4. **Python package** — `src/phasic/`. `__init__.py` (very large, ~13k lines) defines `class Graph(_Graph)`, the primary user-facing object; nearly all user-visible methods hang off it. Everything else in the directory is a supporting submodule.

### The central computation pattern: trace-based elimination

This is the key idea to understand (`trace_elimination.py`). Instead of symbolic expression trees (which blow up exponentially), graph elimination is done **once** with unit weights while **recording a linear trace of arithmetic operations** (O(n³) record). That trace is then **replayed** with concrete parameter vectors θ (O(n) per replay). The replay is pure-array and therefore JAX-compatible: `jit`, `grad`, `vmap`, `pmap` all work, and it scales to 100k+ vertices. Any parameterized graph uses this path regardless of caching flags.

### Hierarchical / SCC caching and distribution

For large graphs (`hierarchical_trace_cache.py`, `distributed_scc.py`): the graph is decomposed into strongly connected components, each SCC's trace is computed in parallel and **hash-deduplicated**, then results are stitched back in topological order. Parallelism runs via JAX vmap/pmap, OpenMP (C composer), or across SLURM nodes. Related: `graph_cache.py` / `trace_cache.py` (on-disk caches under `~/.phasic_cache/`), `cloud_cache.py` + `compute_repository.py` (IPFS-published trace registry), `srun_magic.py` / `cluster_configs.py` / `parallel_utils.py` (SLURM). The `test_slurm_*` and `test_scc_*` tests cover this.

### Inference & likelihood entry points

Parameter inference methods are `Graph` methods: `svgd()` (Stein Variational Gradient Descent — the primary method, implemented in `svgd.py` / configured by `svgd_config.py`), `mcmc.py`, `method_of_moments.py`, `probability_matching.py`. The `Graph.pmf_from_graph*` / `moments_from_graph` classmethods build **JAX-differentiable** likelihood callables (routing through `ffi_wrappers.py` and the trace machinery). `bffg.py` implements backward-forward-backward-Gibbs / importance-weighting on sampled paths.

### Parameterized edges & weight modes

Edges can carry coefficient vectors and be re-weighted per θ via `update_weights(theta)`. Weight modes: **linear/log** (dot product of edge coefficients with θ — lengths must match), or **callback/formula** (`weight_formula.py`, a small expression language over `t0..` = θ and `c0..` = full coefficients, allowing auxiliary data in coefficients beyond θ). `theta_dim` is the parameter dimension; for formula/callback modes it is resolved at inference time rather than at construction.

### State indexing

`state_indexing.py` provides `StateIndexer` / `Property` / `PropertySet`: a mixed-radix bijective mapping between flat integer state indices and structured lineage-property dicts, so population-genetics state spaces can be defined at runtime instead of via hard-coded structs.

### Configuration

`config.py` — explicit configuration with **no silent fallbacks**; features/backends must be enabled deliberately. Use `phasic.configure(...)` and `phasic.get_available_options()`. Note `OMP_NUM_THREADS` is auto-detected and set *at import time before the native module loads* (OpenMP reads it on library load); pre-set it in the shell to override. JAX is forced into 64-bit mode at import because the FFI requires F64 buffers. JAX-dependent symbols are lazily imported via the package-level `__getattr__`.

## Release workflow

Version lives in `pyproject.toml`. `pixi run bump [patch|minor|major]` bumps and commits; `pixi run release` tags and triggers conda/pypi builds; `pixi run version` chains test → docs → bump → release. Docs are Quarto/quartodoc (`pixi run docs`, `pixi run api`).
