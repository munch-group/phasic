# De-risk findings — one C++-owned discretize() (phase −1)

Plan: `discretize-cpp-ssot-plan.md` §7 phase −1. Branch
`derisk/discretize-cpp`, worktree `/Users/kmt/phasic-discretize` with its
own pixi environment. Measured 2026-10-13 on macOS, 16 GB RAM, every run
through the memory-capped runner (6 GB), one at a time, with `HOME` pointed
at a scratch directory so the graph cache could not touch the user's cache.
Scripts: `experiments/dr_discretize_cpp_sizes.py`,
`experiments/dr_discretize_cpp_island_sizes.py`,
`experiments/dr_discretize_cpp_costs.py`,
`experiments/dr_discretize_cpp_bitident.py` (+ fixture
`experiments/_dr_cpp_discretize_fixture.cpp`).

## (a) Where discretize() spends its time today

Parameterized graphs; one run each; times in seconds.

| Graph | V | E | Python widening | discretize, fixed rate | discretize, callback | aux loop (≈ total − widening − normalize) | normalize | C clone() |
|---|---|---|---|---|---|---|---|---|
| coalescent n=40 | 37,339 | 432,938 | 1.82 | 2.54 | 2.63 | 0.72 | 0.003 | 2.99 |
| two-island n=20 | 24,843 | 246,917 | 3.96 | 4.37 | 4.56 | 0.41 | 0.001 | 1.10 |
| two-island n=22 | 49,011 | 527,359 | 10.80 | 11.58 | 12.16 | 0.78 | 0.002 | 4.55 |
| coalescent n=50 | 204,227 | 2,977,225 | 13.27 | 17.72 | 18.28 | 4.43 | 0.022 | **109.9** |

- **The Python widening loop is 70–95% of `discretize()`'s time** on
  parameterized graphs: 4 µs per vertex+edge on the coalescent, 15–19 µs
  on the two-island model (longer states). Moving it into C++ is where the
  gain is.
- **The aux-vertex loop is small** (0.4–4.4 s) and already mostly native
  (`add_aux_vertex` is C++).
- **`normalize()` is negligible** (≤ 0.02 s).
- **Model families differ** as expected (CLAUDE.md: never quote one
  exponent): per element, the two-island model costs 4× the coalescent in
  the Python loop.

## (b) Cost of calling a Python rate callback per vertex

- Inside today's Python `discretize`: callback minus fixed rate is
  2.3–2.7 µs per vertex (coalescent) and 7.6–11.9 µs per vertex
  (two-island, longer state arrays), i.e. 0.1–0.6 s per call at these
  sizes.
- Proxy for a C++ → Python call per vertex through pybind: `Graph(callback)`
  already does exactly that (pybind's callback builder calls the Python
  model callback once per vertex from C++); it is dominated by the model
  callback's own work (107–151 µs per vertex for the coalescent,
  1.3–1.6 ms for the two-island model), so the crossing itself is not
  measurable against it. No GIL problem: the builder works today.
- **Conclusion:** with a Python rate callback, a C++ `discretize` keeps one
  Python call per transient vertex; that costs well under a second at
  200k vertices. Not a reason to change the design.

## (c) Bit-identity of the C++ and Python discretize

The existing C++ `phasic::Graph::discretize` (constant graphs only) against
Python `discretize` on the same constant coalescents (n = 6, 12, 30;
12–5,605 vertices), fixed rate 0.1 and callback 0.3·Σstate:
**bit-identical** vertex states, `is_aux` flags, per-vertex edge lists
(target and weight, in order) and `rewards`, compared exactly. The only
difference is the known one: C++ does not set `was_dph` (Python does).
Both call the same C++ `add_aux_vertex` and `normalize`, so this is
expected and now measured.

Also confirmed: a Python `Graph` can be passed into a cppimport-compiled
C++ function taking `phasic::Graph&`, and the returned `phasic::Graph`
wraps back with `Graph(...)`. The phase-3 gate ("drive the phase-0
records through C++ directly") is feasible as designed.

## New finding — C `ptd_clone_graph` is O(V·E)

`ptd_clone_graph` (`src/c/phasic.c:1284-1294`) finds each edge's target
position by scanning the whole vertex array, once per edge. Measured: 3.0 s
at 37k vertices, 110 s at 204k (time ratio 37× for a 5.5× size increase,
≈ quadratic). Every vertex already stores its position in
`vertex->index`, so the scan can be a direct lookup.

It affects every clone: Python `Graph.clone()`/`copy()`, `discretize()` on
constant graphs, `laplace_transform()`, and the private clone the exact
gradient code takes. At the 200k–600k scale target this is minutes per
clone.

Not fixed here: it is existing C code outside this plan. The new C++
`widen_layout` must look targets up by `vertex->index` (never by scan),
so it does not inherit the problem. Recommended as a separate small fix
(decision for Kasper).

## Re-evaluation of the plan

- Phases 0–5 stand as planned.
- Phase 1 gains two requirements: `widen_layout` resolves edge targets by
  `vertex->index`; and a performance gate: faster than the Python helper on
  the coalescent n=40 and two-island n=22 fixtures, and roughly linear
  from n=40 to n=50 (memory-capped).
- No change from (b) or (c).

## Found while pinning (phase 0, 2026-10-13)

- **`discretize()` of a native discrete graph is broken** (existing
  behaviour, pinned as-is in the phase-0 records): the returned `rewards`
  vector is one entry shorter than the graph (5 vertices, 4 rewards), so
  `expectation_discrete(rewards)` raises, and the distribution changes
  (pdf(2) goes from 0.21 to 0). Likely cause (not verified): the discrete
  `normalize()` adds a vertex after `rewards` is computed. Candidate for a
  refusal ("discretize() expects a continuous graph").
- **A rate callback receives a writable `int32` numpy array** (the vertex
  state). Pinned as frozen API. Mutating it is harmless: `Vertex.state`
  returns a fresh copy (read from source, `phasic_pybind.cpp:4377-4383`:
  a new `std::vector` wrapped in a numpy array). A C++ discretize must keep
  passing a copy.

## Phase 1 performance gate (2026-10-13)

`experiments/dr_widen_layout_perf_gate.py`, memory-capped: C++ widening vs
the Python helper on the same graphs: coalescent n=40 0.13 s vs 1.82 s
(14×); two-island n=22 0.31 s vs 10.8 s (35×); coalescent n=50 0.96 s vs
13.3 s (14×). Cost per vertex+edge 0.28-0.53 µs, ratio n=50 / n=40 1.08
(linear). ALL PASS.
