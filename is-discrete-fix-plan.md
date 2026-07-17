# Fix plan — `is_discrete` propagation + discrete-moment correctness

Branch: `fix/is-discrete-propagation` (off `master` `0754c66f`). These are **pre-existing** bugs
(documented as F-008..F-011 on `numerical-refactor-fd-weight-mode`, commit `dec3a244`), fixed here
independently of the FD/weight-mode refactor.

## Root cause (verified)

`is_discrete` is a **Python-only** `Graph` attribute (`__init__.py:1466` default `False`; set by
`discretize()`, `reward_transform_discrete()`, `joint_prob_graph(discrete=True)`). It drives the
Python-level dispatch correctly (`reward_transform` at `:3053` → discrete branch when
`self.is_discrete`), **but `serialize()` drops it** and the C++ `GraphBuilder` has no discreteness
concept. So the entire parameterized/JAX path (`pmf_and_moments_from_graph`) is blind to
discreteness and relies only on a per-call `discrete=` bool that controls the PMF read *only* — not
the reward transform (N1) or moments (N2).

The C `ptd_graph` already carries a `was_dph` latch (`api/c/phasic.h:196`); it just never gets set
on graphs built by `GraphBuilder` because serialize drops the flag.

## Design

Propagate discreteness so the graph self-describes, and make **every** compute path dispatch on
`is_disc = (per_call_discrete || graph_is_discrete)`:

1. `serialize()` emits `is_discrete`.
2. `GraphBuilder::parse_structure` reads it into a member `bool is_discrete_`.
3. `compute_pmf_and_moments` / `compute_moments_impl` / the FFI handlers dispatch reward-transform,
   moments, and PMF on `is_disc`.

Discrete reward transform needs **integer** rewards (`reward_transform_discrete(std::vector<int>)`,
`phasiccpp.h:690`): validate integer-valued, else raise (no silent fallback).

Discrete moments = continuous moments with a **graph-independent** correction (proven: `U=(I−P)⁻¹`
commutes with `P`): `E[N²]_disc = E[T²]_cont − E[T]_cont = m[1] − m[0]`. General order via the
binomial→factorial→Stirling conversion; order 2 (the default) is the priority.

## Batches (test gate between each; build with `pixi run install-dev`; gate on TARGETED tests, never
full suite — pre-existing failures exist)

### Batch 1 — surgical, independent (do first)
- **`variance_discrete`** (`phasic_pybind.cpp:364`): empty-rewards branch returns `m[1]-2*m[0]`
  (= 53.33 for the test DPH); correct is `m[1]-m[0]-m[0]*m[0]` (= 15.556). The **rewards** branch
  (`:376`) is already correct. One-line fix.
- **N4** reward-length validation on `pmf_and_moments_from_graph*` (Python entry, mirroring
  `_validate_rewards`) so a wrong-length reward raises instead of the OOB read in
  `_ptd_graph_reward_transform` (`phasic.c`).
- Tests: `variance_discrete`==closed form on 3 DPHs; wrong-length rewards → raises (no OOB).

### Batch 2 — is_discrete propagation + N1
- `serialize()` (`__init__.py` near `:754`) add `result['is_discrete']`.
- `GraphBuilder::parse_structure` (`graph_builder.cpp:19`) parse `is_discrete_`.
- reward-transform sites: pybind `compute_pmf_and_moments` (`graph_builder.cpp:706,741`) + FFI
  multivariate (`graph_builder_ffi.cpp:791,840`) → discrete transform on `is_disc`, with integer
  validation.
- Tests: `pmf_and_moments_from_graph(discrete=True, rewards)` == `reward_transform_discrete` oracle;
  round-trip through serialize preserves `is_discrete`.

### Batch 3 — N2 discrete moments
- Apply the continuous→discrete correction in `compute_pmf_and_moments`/`compute_moments_impl` when
  `is_disc`. Order 2 exact; decide general vs fail-loud for `nr_moments>2` discrete.
- Tests: `pmf_and_moments_from_graph(discrete=True)` moments == summation oracle across ≥3 DPHs.

### Batch 4 (candidate, confirm) — wire the dead `moments(discrete=True)`
- `__init__.py:2033` calls `super().moments_discrete(...)` which is unbound (AttributeError). Route
  it to the corrected discrete-moment computation, or bind a C++ `moments_discrete`.

## Verification
- Reuse `scratchpad/verify_N_all.py` semantics as regression oracles.
- Establish a pre-change PASS baseline of targeted existing tests (discrete / moments / reward /
  gate) so introduced regressions are distinguishable from pre-existing failures.
