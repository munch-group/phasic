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

## Disabled paths / follow-ups

### `Graph.pmf_from_graph_parameterized` — disabled, needs revival

The **builder-based** (`θ → Graph` *function*) likelihood API `Graph.pmf_from_graph_parameterized` (and its only helper `_create_jax_parameterized_wrapper`, both in `src/phasic/__init__.py`) is **disabled** — it raises `NotImplementedError`; the original implementation is preserved directly below the raise (now unreachable) for revival. It is **unused**: SVGD, the model-selection LRT (`model_selection.py`), and everything else route through the **graph-based** parameterized API (`pmf_from_graph` / `pmf_and_moments_from_graph`, which take a pre-built parameterized `Graph` whose edges carry coefficient vectors). It was also **broken** three ways:

- **bug 5a** — it never calls `_ensure_jax_active()`, so the module-level lazy `jax`/`jnp` are `None` → `AttributeError: 'NoneType' has no 'ShapeDtypeStruct'` (unless another JAX path activated them first).
- **bug 5b** — it hardcodes `jnp.float32` for the `pure_callback` result, but the FFI/native path returns/expects **F64** → `Wrong buffer dtype: expected F64 but got F32`.
- **F-001** — its `discrete=True` C wrapper still calls `g.normalize()` on the raw graph. For a native DPH that continuous normalize collapses the chain to a deterministic walk (and zeroes the gradient) — the same defect fixed as "bug 4" in `pmf_from_cpp` (which documents "NO normalize() here").

**Revival checklist:** (1) call `_ensure_jax_active()` at the top of the returned model fn (fix 5a); (2) declare the `pure_callback` result dtype as `times.dtype` / F64, not `jnp.float32` (fix 5b); (3) delete the `g.normalize()` in the discrete `compute_dph_pmf_from_arrays` wrapper, or reject row-sum > 1 (fix F-001, mirroring `pmf_from_cpp`); (4) un-skip and strengthen its tests in `tests/pytest/inference/test_jax_integration.py` (`TestPMFFromGraphParameterized`, `TestJAXGradients`, and `test_jit_parameterized` / `test_jit_with_grad` / `test_vmap_over_parameters` / `test_vmap_nested`, currently `@pytest.mark.skip`) so they assert **values** against a native oracle (they only checked `pmf.shape` before, which is why the normalize bug went unnoticed); add a discrete cross-path gate (`pmf_from_graph_parameterized` == `pmf_from_cpp` == FFI, vs a NegBinomial closed form) on a row-sum≠1 graph. Only worth doing if the builder-function style is actually wanted — otherwise the graph-based API fully covers it.

### B3 exact moment gradient (`exact_moment_grad`, now default `True`) — known gaps, flagged not fixed

`Graph.pmf_and_moments_from_graph`'s exact reverse-mode moment-vector adjoint
(continuous + discrete/was_dph, `weight_mode='linear'`) defaults to `True`
as of commit `f89b5b2b`; FD is used (and logged at INFO) only when out of
scope or explicitly requested. Found via adversarial review of that
default-flip (three independent review passes tasked with refuting, not
confirming); two real bugs surfaced there were fixed (rewards silently
ignored by the exact Jacobian, commit `315ce9c8`; a gradient-norm-clip
defect in `svgd_step` that could crush healthy particles' gradients when a
majority of a batch diverges simultaneously, commit `839a6400`). The
following were flagged during that same review but judged lower-severity /
out of scope for that pass:

- **fwd/bwd inconsistency at rate-blowup.** When the primal hits the
  existing `_rate_blowup_penalty` (theta implies an uncomputable rate; the
  forward returns a fixed 0-moments penalty instead of the real PMF/moments),
  FD correctly differentiates through that penalty (its probes re-evaluate
  the same fail-soft forward, so the FD slope reflects the penalty), but the
  exact path has no equivalent guard — `ptd_moments_grad_theta` /
  `ptd_moments_grad_theta_dph` just compute the true analytic Jacobian of
  the (never-computed) real moments, as if the penalty hadn't fired. No
  test currently exercises this combination (exact_moment_grad=True at a
  theta past the rate-blowup threshold).
- **Unguarded "slow band" in continuous PDF cost.** Between a "normal" rate
  and the much higher threshold where `_is_rate_blowup`/the native step-cap
  actually fires (~2.5e8, `src/phasic/__init__.py` `_RATE_BLOWUP_EXC`
  comment), the continuous PDF's uniformization cost scales with
  rate·granularity·Σt with no guard at all. The `svgd_step` gradient-norm
  clip (`_GRAD_NORM_CLIP_MULT`, `src/phasic/svgd.py`) makes a particle
  landing in this band far less likely (it damps the update that would
  fling a particle there) but does not structurally prevent it — a
  different model/regularization/seed could still land a particle in the
  slow band and pay the cost every iteration thereafter.
- **`Graph.moments_from_graph` and `Graph.method_of_moments` are entirely
  separate FD-only code paths**, untouched by the exact-grad work.
  `moments_from_graph` has its own `custom_vjp` with an unconditional
  central-difference backward (`src/phasic/__init__.py`, no
  `exact_moment_grad` param, no logging). `method_of_moments.py` hands its
  model to `scipy.optimize.least_squares` with no `jac=`, so scipy computes
  its own internal FD Jacobian, independent of `exact_moment_grad`, with no
  visibility either way.
- **`pmf_and_moments_from_graph_multivariate` has no `exact_moment_grad`
  passthrough kwarg.** It composes correctly via per-feature calls to
  `pmf_and_moments_from_graph` (so it inherits the default-on exact path and
  its logging automatically), but a caller cannot force FD directly through
  this entry point — only by editing the underlying 1-D model construction.

None of these are regressions from the default flip (the multivariate/
moments_from_graph/method_of_moments gaps predate it and were never in
scope; the rate-blowup and slow-band gaps are pre-existing robustness edges
that the clip merely makes less likely to hit). Worth a dedicated follow-up
pass: de-risk each fix independently (as with the two bugs already found),
and put every plan and every fix through adversarial review before
merging — the two real defects fixed this session were both found by
review, not by the original implementation or its own tests.
