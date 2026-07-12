# Fix plan — F1 / F2 (+F3) before signing off the numerical refactor

> **STATUS: Batch A is IMPLEMENTED and GREEN.**
>
> | gate | result |
> |---|---|
> | forward parity vs `git HEAD` | **167 bit-identical, 0 value differences, 0 unintended new raises** |
> | `'linear'` FD branch vs the old `nonneg=True` formula | **bit-identical at all 10 probe points** (θ = 0 … 1e-30) — Batch A cannot alter a linear-graph gradient |
> | F1 hole closed | `jax.grad` on a `log` graph is finite at θ = 1e-15 / 1e-16 / 1e-20, where it previously **raised** |
> | new regression tests | 21 pass with `PHASIC_SOURCE_DIR`; 20 pass + 1 skip without |
> | 20-target regression subset (33 min) | **580 passed, 50 skipped, 21 xfailed, 0 failed** |
>
> Skips went 49 → 50: the one added skip is the source-dependent `moments_from_graph` value
> test. Batch B remains **not started** and is explicitly out of scope for sign-off.
>
> The only forward-behaviour changes are the **4 deliberate** new raises
> (`joint_index|mode=log`), plus 18 `moments_from_graph` non-linear cells that already
> failed on HEAD (compile error) and now fail with a clear message instead.

Companion to `audit-phase1-forward-parity.md` (Phase 1: forward parity **confirmed**, 0
regressions). This plan closes the two findings that should block sign-off, plus the F3
sibling that shares F2's root cause.

**Guiding principle (your stated one):** *an API must work as specified or fail — never
silently branch.* Both F2 and F3 today return a **silently wrong number**. Batch A converts
those into loud failures without changing a single currently-correct value. Batch B makes
them actually work, and is *not* required for sign-off.

---

## Scope, precisely (what is and is not broken)

| | broken? | root cause | reaches `svgd()`? |
|---|---|---|---|
| `pmf_from_graph` | ✅ correct for all 4 modes | routes via `GraphBuilder`, which honours `weight_mode` (`graph_builder.cpp:29-36`, `:346`) | — |
| `pmf_and_moments_from_graph` | ✅ correct (mode is live) | has a callback branch | yes — **the SVGD path is fine** |
| `moments_from_graph` | ❌ ignores `log`+`callback`+`formula`; returns **linear** | JIT-codegen bakes a linear dot product into `build_model` | **no internal callers** — standalone public API only |
| `joint_index` | ❌ ignores `log` only (callback/formula OK) | `ComputeSojournTimesFfiImpl` hardcodes `/*use_log=*/false` (`graph_builder_ffi.cpp:887`, `:941`) | yes (`joint_index=True`) |
| daisy handlers | ❌ same `use_log=false` hardcode (`:1528`, `:1782`, `:1827`) | ditto | yes (epoch path) |
| `_fd_probe_points` for `log` | ⚠️ `nonneg=False` on a false premise; residual hole at θ ≤ 1e-15 | comment/`__init__.py:732` | gradients only |

**Mitigations that lower urgency (do not skip reading these):**
- `log` requires **every** `(cᵢ·θᵢ) > 0`, so it is unusable on any zero-coefficient graph —
  i.e. the entire coalescent/joint-prob family. F3 + the daisy hardcode only bite on
  dense-coefficient graphs.
- F2's real blast radius is `formula`/`callback` (mainstream, works on sparse graphs), **not**
  `log`.
- F1's hole is θ ≤ 1e-15 — a collapsed parameter. HEAD broke at θ < 1e-7, so the refactor
  already improved this by 8 orders of magnitude.

---

## BATCH A — blocking for sign-off

Small, raise-only + comment fixes. **No currently-correct forward value changes.**

### A1 (F1) — make the `log` FD probe sign-preserving; kill the false premise

`_fd_probe_points` (`__init__.py:702`) takes `nonneg: bool`, set only for `'linear'`. Replace
the boolean with the graph's weight mode and give `log` its own rule:

- **`linear`** — unchanged: floor the minus-probe at `_FD_MIN_THETA` (θ *is* the rate scale;
  θ=0 makes the moments elimination divide by a zero exit rate).
- **`log`** — **purely multiplicative, sign-preserving**: `θ± = θᵢ·(1 ± _FD_REL_STEP)`, with no
  absolute floor. This never crosses zero at any magnitude and preserves θ's sign (so it is
  correct even if a coefficient is negative, where the product-positivity rule requires a
  negative θ). `denom = 2·_FD_REL_STEP·|θᵢ|`. At θᵢ = 0 the *forward* is already invalid under
  `log` (`c·0 = 0` is non-positive), so no floor is needed or wanted.
- **`callback` / `formula`** — unchanged (no floor): the user's expression may mean anything.

Then fix the false claims: `__init__.py:732` and §3 of `numerical-refactor-handoff-plan.md`
both say θ under `log` "is a log-scale, legitimately negative". Replace with the truth:
*`log` means `weight = Π(cᵢ·θᵢ)` in log-space and requires every product strictly positive —
it is `linear`'s positivity requirement, not a relaxation of it.*

**Test gate (fails today, passes after):**
`jax.grad` of a `log`-mode model at θ=[1, 1e-15] and [1, 1e-16] must be finite.
Currently raises `ValueError: log weight mode requires ... products to be positive`.

### A2 (F2) — `moments_from_graph` must raise, not silently linearise

Add a raise-only guard: if `graph.weight_mode != 'linear'`, raise with a message naming the
mode and pointing at `pmf_and_moments_from_graph` (which *does* honour it).

Safe because `moments_from_graph` has **no internal callers** — `svgd()` never touches it.
No correct result changes; only wrong ones become errors.

**Test gate:** `moments_from_graph` on a `formula`/`callback`/`log` graph raises; on a
`linear` graph it is **bit-identical** to today.

### A3 (F3) — `joint_index` + daisy must raise on `log`, not silently linearise

The five FFI sites hardcode `/*use_log=*/false`. Until they are threaded properly (Batch B),
raise at the Python boundary when `weight_mode == 'log'` for `pmf_from_graph_joint_index`
and `daisy_chain_joint_probs`.

Note this is *currently unreachable* on coalescent-shaped graphs (zero coefficients make
`log` raise at build), so the guard mostly documents an invariant — but it removes the
silent-wrong-answer path on dense-coefficient graphs.

**Test gate:** a dense-coefficient `log` graph through `joint_index` raises instead of
returning the linear answer (`0.625`).

### A4 — correct the written record

`numerical-refactor-handoff-plan.md` §3 (the `log` premise) and §8 (the whitelist must cover
`moments_from_graph` **in full**, not just `use_ffi=False` — `use_ffi` is inert, F4).

### Batch A gate

Re-run the Phase-1 forward-parity harness against HEAD. Expected diff vs the current run:

- all 177 linear cells still **bit-identical** — this is the real gate;
- `moments_from_graph` × (log, callback, formula) and `joint_index` × log flip `ok → raise`
  — **intended and recorded**;
- everything else unchanged.

Then run the 21-target subset from §5 of the hand-off. Do **not** gate on a full-suite green
(pre-existing failures; ~8 h serially).

---

## BATCH B — after sign-off (real support, not guards)

1. **B1** — thread `use_log` through the five FFI sites (`graph_builder_ffi.cpp:887, 941,
   1528, 1782, 1827`). `GraphBuilder` already computes exactly this
   (`graph_builder.cpp:346`: `bool use_log = (weight_mode_ == WeightMode::LOG)`), so it is a
   plumbing change, not a design one. Converts A3's raise into a correct answer.
2. **B2** — make `moments_from_graph` honour all four modes, by routing it through
   `GraphBuilder` (or reusing `pmf_and_moments_from_graph`'s callback branch) instead of the
   linear JIT codegen. Converts A2's raise into a correct answer.
3. **B3** (F4/F5) — dead flags: `moments_from_graph(use_ffi=…)` selects nothing
   (`:6441`/`:6451`); `pmf_from_graph(use_cache=…)` is never read yet its docstring promises
   caching (`:3501`, `:3551`). Wire up or delete — including the docstrings.
4. **B4** (F6) — `daisy_chain_t_eval` is inert under the **default** `final_read='sojourn'`,
   and `svgd()` still runs the expensive `_resolve_daisy_chain_t_eval` probe and discards it.
   Raise, or document + skip the probe.
5. **B5** (F7) — a non-start vertex whose state is the all-zero vector makes
   `moments_from_graph` return `inf` / "NaN at vertex 0" while `Graph.moments()` is fine.
   Guard or document.

---

## Recommendation

Do **Batch A** and sign off. It changes no correct value, converts three silent-wrong-answer
paths into loud errors, and corrects a false premise that would otherwise mislead the next
maintainer. Batch B is genuine feature work and should not gate a refactor whose forward
parity is already proven.

---

## Batch A — as-built notes (things that bit me; read before touching this again)

### The `nonneg` bool → `weight_mode` string rename has a trap

`_fd_probe_points`' third argument used to be `nonneg: bool`. Any *stale* caller passing
`nonneg=True` would now compare unequal to every known mode and **silently fall through to
the no-floor branch** — precisely the hazard the helper exists to remove. The refactor's own
`test_daisy_chain_c_path.py::test_fd_backward_never_probes_a_negative_rate` was such a
caller, and it is how this was caught.

`_fd_probe_points` therefore now **raises** on an unrecognised mode. Allowed values:
`'linear'`, `'log'`, `'callback'`, `'formula'`, or `None` (opaque user semantics, used by
`pmf_from_cpp`). Do not re-introduce a boolean.

### The `'linear'` branch is provably unchanged

Verified bit-for-bit against the old `nonneg=True` formula at θ ∈ {1.0, 2.0, 1e-4, 1e-7,
1e-8, 1e-9, 1e-12, 1e-30, 0.0, 5.5} — all 10 probe triples identical. **Batch A cannot alter
any gradient on a linear graph.** That is what rules it out as the cause of any SVGD
convergence wobble.

### Two test-suite facts

1. **`inference/test_svgd_correctness.py::test_basic_convergence` is FLAKY / order-dependent.**
   It failed in one 20-target subset run and passed in the next, and passes in isolation
   (reported mean error 0.290 against a 15% threshold). It uses a **linear** graph, so by the
   bit-identity result above Batch A cannot have caused it. It is **not** in
   `failing_tests.md`. Someone should either tighten its seed or widen its tolerance — but it
   is a pre-existing property of the SVGD convergence check, not of this change.
2. **The `moments_from_graph` value test is SOURCE-DEPENDENT and SKIPS by default.**
   `moments_from_graph` JIT-compiles C++, so any test that reaches its *value* needs
   `PHASIC_SOURCE_DIR` (a non-editable install is a copy under `site-packages`, where no
   `src/` exists). It is guarded with a `requires_sources` skipif.
   The **guard** tests (A2/A3) need no source — the weight_mode check fires *before* the
   compile — so they run in a default suite. Run with
   `PHASIC_SOURCE_DIR=/Users/kmt/phasic` to exercise the skipped one.

### Gate subset used

20 targets (`test_svgd.py`, `inference/`, the daisy/epoch files, the joint-index /
weight-formula / gate files, `test_method_of_moments.py`, `test_mcmc.py`,
`test_svgd_config.py`) + the new `test_weight_mode_probe_and_guards.py`. ~33 min.
A full-suite green was **not** used as a gate (~8 h serially; pre-existing failures).
