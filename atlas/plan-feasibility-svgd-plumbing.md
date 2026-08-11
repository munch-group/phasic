# Feasibility/scoping: SVGD exact-grad plumbing, joint-index baked mode, two mechanical fixes

Investigation date 2026-08-05. Starting hypotheses were `atlas/exact-fd-atlas-svgd-reachability.md`
and `atlas/exact-fd-atlas-python-wiring.md` (both dated 2026-08-01/04) plus
`b3-joint-index-plan.md`. Every claim below was re-verified against the CURRENT
checked-out source (`master`, HEAD `cadf1ca4` at investigation time — one commit
past what the conversation's initial git-status snapshot showed) by direct
`Read`/`grep`, not by trusting the prior docs. Line numbers are
`src/phasic/__init__.py` unless stated otherwise.

**Important housekeeping finding, upfront:** `b3-joint-index-plan.md`'s "D6 —
`lax.cond`/`vmap` redesign" batch (the thing that would let `exact_grad`
default back to `True` and actually skip FD under `vmap`) is **planned and
de-risked only** (`4f5936d0`, `7633a895`) — **not implemented**. Confirmed by
grepping current source for `_jix_probed_ok` (the D6 design's construction-time
probe latch): zero matches. `pmf_from_graph_joint_index` today is still exactly
the D5 state: `exact_grad: bool = False` (`:7656`), `jax.lax.cond` at `:8268`,
static `_baked` exclusion at `:7892-7898`. The most recent commit on this file
(`cadf1ca4`, "docs: exact/FD gradient atlas") is explicitly a retrospective on
why the D6 work almost got built on a false premise — `Graph.svgd()` never
reaches `pmf_from_graph_joint_index` with `exact_grad` at all, so D6 was
solving a problem with no real caller yet. That context directly motivates
this document's sub-investigation A.

---

## A. `Graph.svgd()` plumbing

### A.0 Confirmed current signature and dispatch

`Graph.svgd()` signature: `:5241-5282`, ~42 keyword parameters, read in full —
no `exact_grad`, `exact_moment_grad`, or similarly-named parameter anywhere.
Dispatch tree confirmed by direct read (`:5920-6120`), matching the prior
atlas's line numbers almost exactly (they were accurate):

1. **Joint-prob graph + `epoch_starts` set → daisy-chain** (`:6017-6070`,
   `self._daisy_chain_svgd_model(...)`).
2. **Joint-prob graph, no `epoch_starts` → `pmf_from_graph_joint_index`**
   (`:6093-6097`), with `_bake_obs = observed_data if exposure is None else None`
   (`:6092`) — i.e. baked/dedup mode is the **default**, exposure-set is the
   exception.
3. **Non-joint, 2D rewards → `pmf_and_moments_from_graph_multivariate`**
   (`:6104-6108`).
4. **Non-joint, 1D rewards → `pmf_and_moments_from_graph`** (`:6111-6114`).
5. **Non-joint, no rewards → `pmf_and_moments_from_graph`** (`:6117-6120`).

`SVGD.__init__` (`src/phasic/svgd.py:4948-4975`, full signature read) also has
**no** `exact_*` kwarg. It takes an opaque `model` callable and never inspects
its gradient method. The gradient itself is always `jax.grad(self._log_prob)`
(`svgd.py:6394`) or, for the regularized/unified path,
`jax.grad(log_prob_fn)` built via `_precompile_unified`
(`svgd.py:6434` — the actual name is `_precompile_unified`/`_precompile_model`,
not `_precompile_gradient`; the task's phrasing was approximate, verified by
grep — no `_precompile_gradient` symbol exists anywhere in `svgd.py`).
Critically, `self.rewards = rewards` is fixed once at construction
(`svgd.py:5702`) and baked into every gradient call via
`partial(self._log_prob_unified, ..., rewards=self.rewards)`
(`_create_log_prob_fn_with_regularization`, `svgd.py:6426-6432`) — read in
full, confirms the prior atlas's claim exactly.

**Verdict on the "does SVGD's own internals need the kwarg too" question: No.**
`SVGD` never constructs a model and never branches on exact-vs-FD; it only
consumes whatever closure `Graph.svgd()` (or a direct caller) hands it via
`jax.grad`. This is **purely a `Graph.svgd()`-level (model-construction-time)
concern** — confirmed, not assumed, by reading `_precompile_model`/
`_precompile_unified`/`optimize` and finding no gradient-method-aware code
anywhere in `svgd.py`.

### A.1 Per-leaf scope of what plumbing `exact_grad`/`exact_moment_grad` would require

| Leaf | What a kwarg would need to do | Genuinely simple plumbing? |
|---|---|---|
| 1. Daisy-chain | **Nothing to plumb** — `_daisy_chain_svgd_model` (`:4254-5056`) has two `@jax.custom_vjp` sites (`:4669`, `:4900`), both explicitly FD-only ("custom_vjp: FD backward, skipping fixed indices", `:4668`; "custom_vjp wraps `_per_obs_core` to provide the FD backward", `:4895`). Grepped the full function body for `exact_grad`/`exact_moment_grad`: zero matches. There is no exact-gradient implementation to select. | **No — blocked, not a plumbing task at all.** Depends on batch (d) (see Cross-batch section). |
| 2. Joint-index (baked, the common case) | Passing `exact_grad=True` through would still hit the static `_baked` exclusion (`:7892-7898`) and fall back to FD — **the kwarg alone buys nothing** for the default (`exposure=None`) sub-case. | Plumbing itself is trivial (one new `Graph.svgd` param, one pass-through at `:6093-6097`), but **low-value alone** — needs sub-investigation B (baked-mode support) to matter for the common case. |
| 2b. Joint-index + `exposure` set | `_bake_obs=None` here (`:6092`), so the static baked exclusion does *not* fire. Plumbing `exact_grad=True` through would make this sub-case structurally reachable, subject only to the graph's own `weight_mode`/`was_dph` gates. | **Yes — simple, and immediately useful** for this sub-case specifically. No dependency on (a)/(d)/B. |
| 3. Multivariate (2D rewards) | `pmf_and_moments_from_graph_multivariate` (`:8283-8286`, full signature read) has **no `exact_moment_grad` parameter to plumb into** — it would need a new kwarg added to that function's own signature first, forwarded to its internal `model_1d = cls.pmf_and_moments_from_graph(...)` call (`:8375-8378`, currently passes no exact kwarg). Even with that plumbing, every per-feature call passes a concrete non-`None` `reward_j` (`:8420`/`:8448` per the prior atlas, confirmed structurally by reading `model_1d(theta, times_j, rewards=reward_j)` call sites), which trips `pmf_and_moments_from_graph`'s own dynamic rewards guard (`_rewards_provided` at `:7559`, forces FD at `:7561-7566`) on **every single call**. | **Plumbing is easy; it is dynamically useless without batch (a).** Two-layer fix needed: add the kwarg to the multivariate function AND land (a)'s rewards support in the underlying exact Jacobian. |
| 4. 1D rewards | `pmf_and_moments_from_graph` already has `exact_moment_grad` (default `True`, `:6826`) — plumbing is a one-line pass-through at `:6111-6114`. But `SVGD.rewards` is fixed non-`None` for the whole run (`svgd.py:5702`) and passed on every call (`svgd.py:6431/6207`) → the same `_rewards_provided` guard (`:7559-7567`) fires every gradient step, **regardless of the kwarg's value**. | **Plumbing is trivial; it is dynamically useless without batch (a).** Single-layer (no multivariate wrapper in the way), but same rewards-guard blocker. |
| 5. No rewards | `pmf_and_moments_from_graph` — plumbing is a one-line pass-through at `:6117-6120`. `rewards` is always `None` on this branch, so the guard never fires. Already reachable today via the callee's own default (`True`). | **Yes — trivial AND already working today with zero code change** (this is the one leaf the prior atlas correctly identified as reachable now). Plumbing the kwarg only adds the ability to explicitly force `exact_moment_grad=False`, which isn't currently possible through `Graph.svgd()`. |

### A.2 Headline verdict for A

Adding a top-level `exact_grad`-style kwarg to `Graph.svgd()` is mechanically
simple **as plumbing** — five `if`/`elif` branches, five one-line
pass-throughs, no new control flow. But its **value is leaf-dependent**:

- **Leaf 5** (no rewards): already works; plumbing only adds an explicit
  opt-out, not new capability.
- **Leaf 2b** (joint-index + exposure): plumbing alone unlocks real value,
  no other batch needed.
- **Leaf 2** (joint-index, baked/common case): plumbing is necessary but not
  sufficient — needs sub-investigation B's baked-mode extension to matter.
- **Leaves 3 and 4** (rewards-bearing): plumbing is necessary but not
  sufficient — needs batch (a) (rewards support in the moments adjoint) to
  matter at all; without it, the kwarg would silently do nothing (which,
  given this codebase's no-silent-fallback ethos, argues for **not** exposing
  the kwarg on these leaves until (a) lands, or exposing it with an explicit
  warning/log that it's a no-op today — a design choice worth deciding
  explicitly rather than shipping a kwarg that appears to work but doesn't).
- **Leaf 1** (daisy-chain): plumbing is impossible/meaningless until batch (d)
  creates an exact-gradient implementation for that path at all.

No wrinkle exists on the `SVGD`-class side for any leaf — confirmed above,
`SVGD` is gradient-method-agnostic by construction.

---

## B. Joint-index baked/dedup mode's exact-path exclusion

### B.0 What baked mode currently does (forward + FD backward)

Read in full (`:7822-8158` forward dispatch, `:8191-8270` `model_bwd`).

Construction time (`:7831-7845`): `_uniq_idx_np, _inverse_idx_np = np.unique(observed_indices, return_inverse=True)`
— standard NumPy dedup + inverse-gather-index. `_inverse_idx_jnp` is the
"which unique-slot does observation `i` come from" map, shape `(n_obs,)`,
values in `[0, n_unique)`.

**Forward** (`:8113-8135`, non-callback weight mode; the callback-mode
analogue is `:8011-8039`): computes `uniq_sojourn`/`all_sojourn` at the
`n_unique`-sized index set (cheap), then **gathers** back to the full
`n_obs` shape: `sojourn_probs = uniq_probs[_inverse_idx_jnp]` (`:8130`,
`:8037`). This gather is exactly why baked mode is fast: the FFI/host-callback
inner loop scales with `k = n_unique`, not `k = n_obs`.

**FD backward** (`_fd_theta_bar`, `:8198-8215`): calls `_compute_pure(theta_±eps, vertex_indices)`
— i.e. it **re-runs the entire forward closure, including the gather step**,
at each perturbed theta, and takes the central difference of the resulting
already-`n_obs`-shaped output against `g_visits` (also `n_obs`-shaped). This
is why FD needs **no explicit reasoning about the gather/scatter at all**: FD
treats the whole model (dedup, gather, quotient rule, everything) as an
opaque black-box function of theta and just numerically differentiates its
observable output. This is the mechanical reason baked mode "just works" under
FD today with zero special-casing.

### B.1 What the exact path does today (non-baked) and why it can't do the same trick

The exact callback `_exact_sojourn_jac_np(theta_np, vertex_indices_np)`
(`:7924-7961`) computes the sojourn-gradient Jacobian **at whatever index set
is passed in** — `union_idx = np.union1d(vi, _jix_all_terminal_np)`
(`:7945`), one C call over the union, then `searchsorted`-gathers `J_obs`/`J_all`
out of the union result (`:7959-7961`). In `model_bwd`'s exact branch
(`:8241-8266`), `vi = _vi_norm` is the **runtime** `vertex_indices` argument —
for the non-baked model this is the full `n_obs`-length array (whatever
`observed_data` was, one entry per observation, with repeats). The Jacobian
`J_obs` therefore has one row **per observation, including duplicates** — the
exact path is not itself using dedup, unlike the forward's baked path. This is
exactly why baked mode is currently a hard exclusion for the exact side
(`:7892-7898`): plugging the existing exact machinery directly into baked
mode's runtime-ignored `vertex_indices` argument would either (a) silently
compute a Jacobian at the wrong (full, non-deduped) index set — defeating the
whole point of baking — or (b) require deriving the correct de-duplicated
version, which is what this section verifies.

### B.2 The claimed fix, verified by derivation

`b3-joint-index-plan.md`'s exact phrase (confirmed present, "Scope this
batch" section): baked mode "would need a scatter-add of the upstream
cotangent by the inverse-index map before the quotient rule — mechanically
not hard, but another independent piece of work." I derived the concrete math
to check this claim rather than taking it on faith:

Forward: `sojourn_probs[i] = uniq_probs[inverse_idx[i]]`, a **gather**.
The VJP of a gather is a **scatter-add** of the downstream cotangent back
into the pre-gather (unique-index) space:

```
g_uniq[u] = sum_{i : inverse_idx[i] == u} g_visits[i]
          = scatter_add(g_visits, inverse_idx, size=n_unique)
          # one line: jnp.zeros(n_unique).at[inverse_idx].add(g_visits)
```

Then the **existing** quotient-rule math (`:8257-8260`, unchanged) applies
verbatim, just evaluated at the unique-index granularity instead of the
full-observation granularity:

```
d_probs_uniq[u, j] = (J_uniq_obs[u,j]*norm - uniq_obs_sojourn[u]*dnorm[j]) / norm**2
theta_bar[j] = sum_u g_uniq[u] * d_probs_uniq[u, j]     # = d_probs_uniq.T @ g_uniq
```

Where `J_uniq_obs`/`uniq_obs_sojourn` are obtained by calling the **existing**
`_exact_sojourn_jac_np(theta, _uniq_idx_jnp)` and the **existing**
`compute_sojourn_times_ffi(structure_dict, theta, _uniq_idx_jnp)` with the
STATIC `_uniq_idx_jnp` (already computed and closed over at construction,
`:7837`) in place of the runtime `_vi_norm` — no new callback signature, no
new C code, no new FFI call shape. This is the "before the quotient rule"
wording's precise meaning: the scatter-add (mapping `g_visits` → `g_uniq`)
happens to the **cotangent input**, upstream of applying the (unchanged)
quotient-rule Jacobian contraction.

**This derivation checks out as correct and genuinely small**: it reuses
100% of the existing exact-path machinery (the C function, the callback, the
quotient rule), swaps one runtime array (`_vi_norm`) for one static array
(`_uniq_idx_jnp`) already sitting in the closure, and adds exactly one new
op (a `.at[idx].add()` scatter-add — a single JAX primitive, not custom
code). `_fd_theta_bar` needs **zero changes** (already correct for baked
mode, per B.0). The plan doc's characterization "mechanically not hard" is
verified, not just trusted.

### B.3 Concrete required changes (scope, not implementation)

1. Remove/relax the unconditional `elif _baked: _jix_exact_enabled = False`
   static exclusion (`:7892-7898`) — replace with a design that supports
   baked+exact (see B.4 for the recommended shape).
2. In `model_bwd`'s exact branch (`:8241-8266`), add an `if _baked:` fork:
   - call the small FFI sojourn calls at `_uniq_idx_jnp` instead of `_vi_norm`
     (mirrors the forward's own `_obs_forward`/`_all_forward` calls, which
     already use `_uniq_idx_jnp` — `:8080`, `:8092`);
   - call `_exact_sojourn_jac_np(theta, _uniq_idx_jnp)` instead of
     `_exact_sojourn_jac_np(theta, _vi_norm)`;
   - compute `d_probs_uniq_exact` via the existing quotient-rule formula at
     `n_unique` granularity;
   - scatter-add `g_visits` into `g_uniq` via `_inverse_idx_jnp` (one new
     line);
   - contract `d_probs_uniq_exact.T @ g_uniq` → `theta_bar`.
3. `fixed_mask` handling is unchanged (applies to the final `theta_bar`,
   orthogonal to indexing granularity — same as today, `:8261-8266`).
4. Update the docstring (`:7707`, `:7854-7860`) and the no-silent-fallback
   log message (`:7892-7898`) since baked mode would no longer be an
   unconditional decline.
5. New tests: exact-vs-FD parity specifically for baked mode (the existing
   `test_exact_grad_joint_index.py` per the plan doc's D5 batch did not cover
   this, since baked mode was out of scope there).

### B.4 Interactions flagged by the task — verified, not assumed

**Does this interact with the `lax.cond`/`vmap` redesign (D6)?** Yes, and
this is the most important finding of sub-investigation B. D6 (not yet
implemented, see housekeeping note above) exists specifically because
`jax.lax.cond` cannot skip a branch once its predicate is batched (always
true under SVGD's real `vmap(grad(loss))(particles)` usage, per D6's
build-free de-risk `experiments/dr_lax_cond_vmap_derisk.py`, 3/3 claims
confirmed per the plan doc). If baked-mode exact support is bolted onto the
**current** `lax.cond`-based wiring (`:8268`), it inherits the exact same
defect: under real SVGD usage, enabling `exact_grad=True` for a baked model
would pay FD **and** exact on every gradient step, not FD-only-when-needed —
i.e., building baked-mode support on top of the current wiring produces
*more* code subject to a known-defective pattern that D6 already has a
designed (though unimplemented) fix for. Two sequencing options:
  - (i) do D6 first (the static-`if`/probe redesign), then add baked-mode
    support directly into the corrected wiring — avoids writing
    `lax.cond`-based code that would need to be revisited;
  - (ii) do baked-mode support first using `lax.cond` (matching today's
    non-baked pattern for consistency), accept the known `vmap` inefficiency,
    and let a later D6-equivalent pass fix both non-baked and baked wiring
    together.
  Given D6's own design doc already treats the redesign as index-set/mode
  agnostic (the probe-and-commit pattern doesn't care whether the index set
  is baked or not), **option (i) is very likely the lower-total-effort
  path** — but this is a genuine sequencing call, not a forced conclusion,
  since (ii) is also viable if baked-mode correctness is wanted sooner than
  the D6 redesign.

**Does this interact with the union-index computation inside the host
callback?** Yes, in a way that reveals a **free optimization**, not a
blocker: today, `union_idx = np.union1d(vi, _jix_all_terminal_np)` (`:7945`)
is recomputed **inside the callback on every gradient call**, because `vi`
(the runtime `vertex_indices`) is a genuine per-call value for the non-baked
model. For baked mode, `vi` would become the **static** `_uniq_idx_np` —
doubly static together with `_jix_all_terminal_np` (also fixed at
construction) — so `union_idx`, `obs_pos`, and `all_pos` could all be
hoisted to construction time and closed over, exactly as `b3-joint-index-plan.md`'s
original "Wiring" section already does for the general (non-baked) design.
This isn't required for correctness (the callback would still work if this
union were recomputed every call, just slightly wastefully) but is a natural
efficiency win specific to baked mode, worth doing in the same batch since
it falls out of the same code change.

**No other structural blocker found.** `custom_vmap` (used on the forward's
`_obs_forward`/`_all_forward`, `:8077-8111`) lives entirely in the forward
path and is untouched by this change — `model_bwd` is a plain function under
the outer `@jax.custom_vjp`, not itself wrapped in `custom_vmap`.

---

## C.1 `Graph.moments_from_graph`'s vmap crash

### Confirmed root cause

`moments_from_graph` (`:6594-6819`). The host-callback function
`_compute_moments_pure(theta_flat)` (`:6767-6779`) does:

```python
theta_np = np.asarray(theta_flat, dtype=np.float64)
...
lib.compute_moments(theta_np.ctypes.data_as(...), len(theta_np), nr_moments, output_np...)
```

with **no `ndim` check at all**. Compare the three siblings, all read in
full:

- `_exact_moments_jac_np` (in `pmf_and_moments_from_graph`, `:6996-7040`):
  `if th.ndim == 1: return _one(th)` else a Python `for _b in range(th.shape[0])`
  loop calling `_one(th[_b])` and stacking into a preallocated `(batch, K, P)`
  array (`:7035-7040`).
- `_compute_pmf_and_moments_cached` (`:7291-7337`): `if theta_np.ndim == 2:`
  loop over `for theta_single in theta_np`, appending results, `np.array(...)`
  at the end (`:7294-7320`); else the unbatched single-call path.
- `_compute_cdf_zero_cached` (`:7408-...`): same `if theta_np.ndim == 2:`
  loop pattern (`:7415-7432`).

All three pure_callback call sites use `vmap_method='expand_dims'`
(`moments_from_graph`'s own call, `:6786`, uses the same
`vmap_method='expand_dims'`) — under `vmap`, JAX's `pure_callback` contract
for `'expand_dims'` hands the host callback the **batched, concrete NumPy
array** (with a real leading batch axis) and expects the return value to
carry a matching leading axis too; it is the callback's own responsibility to
detect and loop over that axis — JAX does not do this automatically inside
the callback body. `_compute_moments_pure` never checks for it, so under
`vmap` (i.e. under any real SVGD run, since SVGD always vmaps its gradient
over particles), `theta_flat` arrives 2-D, `len(theta_np)` returns the
**batch size** instead of `n_params`, and the ctypes call proceeds with a
wrong `n_params` argument against a mis-shaped buffer — a shape mismatch /
wrong-answer/crash, not merely a slow path.

### Confirmed scope of the fix

**Genuinely confined to `_compute_moments_pure`.** The fix is exactly "add the
same `if ndim==2: loop` pattern" the task hypothesized — verified by direct
comparison, not assumed:

```python
def _compute_moments_pure(theta_flat):
    theta_np = np.asarray(theta_flat, dtype=np.float64)
    if theta_np.ndim == 2:
        out = np.empty((theta_np.shape[0], nr_moments), dtype=np.float64)
        for b in range(theta_np.shape[0]):
            out[b] = _compute_moments_pure(theta_np[b])   # or inline the single-call body
        return out
    output_np = np.zeros(nr_moments, dtype=np.float64)
    lib.compute_moments(theta_np.ctypes.data_as(...), len(theta_np), nr_moments, output_np...)
    return output_np
```

**No deeper change needed.** The underlying `lib.compute_moments` (compiled
C++/ctypes function) is only ever invoked on a genuine single 1-D theta
vector inside the loop, exactly as it is today for the unbatched case — this
matches how all three siblings solve the identical problem purely in Python,
without touching their own compiled backends (`GraphBuilder.compute_pmf_and_moments`,
`_moments_grad_theta*`). No change is needed to `_compute_pure`'s
`result_shape`/`jax.ShapeDtypeStruct` declaration either — `pure_callback`'s
own `'expand_dims'` vmap machinery handles expanding the *declared* shape by
the batch dimension automatically; the siblings' own outer wrapper functions
don't touch their ShapeDtypeStruct declarations for batching either (verified
by reading `_compute_pure` at `:7340-7397`, whose `pmf_shape`/`moments_shape`
are computed once, unbatched, exactly like `moments_from_graph`'s own
`_compute_pure` at `:6782-6786`).

One adjacent, not-strictly-required observation: `moments_fn_bwd`'s FD loop
(`:6799-6816`) uses ordinary JAX array ops (`theta.at[i].add(eps)`,
`jnp.sum`, a static Python `for i in range(n_params)`) — these are vmap-safe
by construction (vmap transparently vectorizes ordinary jnp code; the
batch dimension is only ever "visible" as a concrete array dimension at a
`pure_callback` host boundary). So the bug is isolated to the one host
callback, not spread through the backward pass — confirmed by the same
reasoning that explains why the three siblings' FD loops need no ndim
handling of their own even though they cross the SAME kind of boundary via
their own callbacks (which do have the check).

---

## C.2 `ptd_moment0_grad_theta`'s missing guards

Read in full: `ptd_moment0_grad_theta` (`src/c/phasic.c:10678-10726`, gated
`#ifdef PHASIC_B3_VALIDATORS`) and its shared helper `ptd_dbg_reverse_tape`
(`:10529-10612`, same gate), compared against `ptd_moments_grad_theta`
(`:10738-10881`, unconditional/production).

**Exposure check (relevant to how much this matters):** `grep`-confirmed
`ptd_moment0_grad_theta` has exactly one caller anywhere in the repo —
`phasiccpp.h:550` (`Graph::moment0_grad_theta()`), and that C++ wrapper is
**itself** gated `#ifdef PHASIC_B3_VALIDATORS` (`api/cpp/phasiccpp.h:541-555`,
read in full). No pybind exposure, no Python surface, not referenced in
`tests/` or `experiments/`. This function never runs in a normal (non-validator)
build — matches CLAUDE.md's characterization of it as "validator-only... kept
as a de-risk oracle," confirmed by source, not by trusting the comment alone.

### Guard 1 — MPFR gate: confirmed missing, present in the successor

`ptd_moments_grad_theta` calls `ptd_dbg_tape_needs_mpfr(nm, nc)`
(`:10783-10788`) right after building its local numeric-command arrays
(`nm[]`/`nc`, built at `:10766-10779`), declining (-1, FD fallback) if the
primal would need MPFR (double-precision adjoint would then be inconsistent
with an MPFR forward).

`ptd_moment0_grad_theta` never calls this function. Grepped every call site
of `ptd_dbg_tape_needs_mpfr` in `phasic.c` (`:10783`, `:10960`, `:11221`,
`:11529`) — none are inside `ptd_dbg_reverse_tape` (`:10529-10612`, the shared
helper `ptd_moment0_grad_theta` calls) or inside `ptd_moment0_grad_theta`
itself. **The gate is entirely absent from this code path, not merely
unreachable.**

**Wrinkle found (not a plain copy-paste):** `ptd_dbg_reverse_tape` already
builds the exact `nm[]`/`nc` values needed for the gate, locally, at
`:10547-10563` — so the data is available. But `ptd_dbg_reverse_tape` is a
**shared static helper with two callers**: `ptd_moment0_grad_theta`
(`:10702`) and `ptd_debug_reverse_grad` (`:10626`, a *different* validator).
Inserting the gate call directly into `ptd_dbg_reverse_tape` (the natural,
least-duplicative place, since the data is already there) would change
`ptd_debug_reverse_grad`'s behavior too, not just `ptd_moment0_grad_theta`'s —
a second function beyond the one asked about would be affected. The
alternative — duplicating the small `nm`/`nc` tape-build-and-gate-check
inline inside `ptd_moment0_grad_theta` only, without touching the shared
helper — avoids that side effect at the cost of a small amount of code
duplication (on top of the reverse-tape-skeleton duplication already flagged
in CLAUDE.md as a separate, larger follow-up). Given this project's explicit,
repeatedly-stated preference for additive-only changes over modifying shared,
already-shipped code (`feedback_no_modify_existing`), the **local-duplication
option is the lower-risk one of the two**, even though it is not the
"pure copy-paste" the task framing suggested — this is the concrete wrinkle.

### Guard 2 — `coefficients_length==0` NULL-check: confirmed missing, present in the successor, no wrinkle

`ptd_moments_grad_theta`'s edge→theta contraction loop (`:10850-10870`)
explicitly guards:
```c
if (e->coefficients_length == 0) continue;   // :10868
```
with an in-line comment explaining this protects against a NULL
`e->coefficients` pointer for coefficient-less aux edges (produced by
`add_aux_vertex`/`add_aux_vertex_constant`, i.e. `Graph.discretize()` and
`Graph.joint_stop_prob_graph()`), fixed as part of the discrete/was_dph batch
after being "an always-latent segfault... just never previously exercised."

`ptd_moment0_grad_theta`'s analogous loop (`:10709-10716`) has **no such
check** — it unconditionally dereferences `e->coefficients[j]`
(`:10715`) after only validating `sp.kind`/`sp.byte`/bounds, not
`coefficients_length`. This is the exact same bug class, still present here
because this validator predates (and was never touched by) the fix applied
to its successor.

**This one genuinely is a trivial, self-contained copy-paste**: the fix is
local to `ptd_moment0_grad_theta`'s own loop, touches no shared helper, and
is a single added line (`if (e->coefficients_length == 0) continue;`) —
no wrinkle specific to the K=1 special case.

### Net assessment for C.2

Both guards are real, confirmed-missing defects in a validator-only,
build-gated, single-caller function — low severity (matches CLAUDE.md's own
downgrade of similar issues) since it never runs in a production build. One
guard (NULL-check) is a clean copy-paste. The other (MPFR gate) is *nearly* a
copy-paste but forces a real decision about whether to touch a second,
unrelated validator function (`ptd_debug_reverse_grad`) or accept minor code
duplication to avoid that — worth flagging explicitly rather than promising
"pure copy-paste" for both guards uniformly.

---

## Cross-batch dependency analysis

**Does A depend on batch (a) — rewards support for the moments adjoint?**
**Yes, for leaves 3 and 4 (the two rewards-bearing SVGD leaves), confirmed by
direct source reading, not inferred from the atlas alone.** `pmf_and_moments_from_graph`'s
own dynamic guard (`_rewards_provided` at `:7559`, forcing FD at `:7561-7567`)
fires on every call where `rewards is not None`, and `pmf_and_moments_from_graph_multivariate`
has no exact kwarg of its own AND always calls its inner `model_1d` with
concrete non-`None` per-feature rewards (`:8375-8452`, read in full for the
signature and the `model_1d(...)` call pattern). So plumbing `exact_grad`/
`exact_moment_grad` through `Graph.svgd()` for leaves 3/4 would add a kwarg
that is **dynamically inert** until (a) lands — the SVGD-level plumbing work
has zero payoff on its own for these two leaves. This is a real, structural
dependency, not merely a nice-to-have ordering preference.

**Does the daisy-chain leaf (leaf 1) depend on batch (d)?**
**Yes, confirmed directly** — grepped `_daisy_chain_svgd_model`'s full body
(`:4254-5056`) for `exact_grad`/`exact_moment_grad`: zero matches. Both its
`@jax.custom_vjp` sites (`:4669`, `:4900`) are FD-only by construction, with
in-line comments saying so. There is no exact-gradient variant to select for
this leaf — "plumbing a kwarg through" is not merely low-value here, it is
**impossible** until batch (d) ("PMF/PDF gradient re-derivation + daisy-chain")
creates an exact-gradient implementation for the daisy-chain path in the
first place. `Graph.svgd()`-level work on this leaf cannot even begin before
(d) ships something to plumb into.

**Sub-investigation B (joint-index baked mode) is independent of both (a) and
(d)** — it is its own, separate piece of work (not one of the five lettered
batches (a)-(e)), touching only `pmf_from_graph_joint_index`'s own `model_bwd`
and C-level `_sojourn_grad_theta_subset` callback usage. It has a real,
identified interaction with **D6** (the already-planned-but-unimplemented
`lax.cond`/`vmap` redesign for this same function), as detailed in B.4 —
not with (a)-(e).

Batches (b) formula-mode/reverse-tape refactor, (c) hierarchical/SCC tape
compatibility, and (e) callback+MPFR conditioning-floor were checked against
every leaf/fix in this document and found **not** to intersect: none of the
five SVGD leaves route through formula/callback weight mode by default, none
touch hierarchical/SCC caching in a way this document's fixes depend on, and
neither `moments_from_graph`'s vmap fix nor `ptd_moment0_grad_theta`'s guards
have any callback/MPFR-conditioning-floor dependency (the MPFR gate at issue
in C.2 is the ALREADY-EXISTING `ptd_dbg_tape_needs_mpfr`, unrelated to (e)'s
new adjoint work).

---

## Recommended sequence position

The task anticipated a possible split between "plumbing useful today" and
"plumbing that only helps once other batches land" — **this split is real**,
confirmed above, not just a hypothetical:

**Tier 1 — useful today, no dependencies:**
- C.1 (`moments_from_graph` vmap fix) — self-contained, zero dependencies,
  fixes a real crash.
- C.2's NULL-check guard — self-contained, though low-severity (validator-only,
  never runs in production builds).
- A's leaf-2b plumbing (`joint_index` + `exposure` set): add the kwarg,
  pass it through at `:6093-6097` for the exposure sub-case specifically —
  immediately unlocks a structurally-reachable sub-case with no other batch
  needed.
- A's leaf-5 plumbing (no rewards): low value (already works), but cheap —
  bundle only if an explicit `exact_moment_grad=False` opt-out through
  `Graph.svgd()` is wanted.

**Tier 2 — needs its own investigation-B work first, but self-contained
otherwise:**
- B (joint-index baked-mode scatter-add support) — genuinely tractable per
  the derivation above, but best sequenced **together with or after D6**
  (the `lax.cond`/`vmap` redesign already planned for this same function),
  to avoid writing `lax.cond`-based baked-mode wiring that a subsequent D6
  pass would have to revisit. Once B (and ideally D6) land, A's leaf-2
  (joint-index, the common baked case) plumbing becomes genuinely valuable
  for the first time.
- C.2's MPFR-gate guard — tractable, but requires an explicit decision
  (touch the shared `ptd_dbg_reverse_tape` helper — affecting a second
  validator — vs. accept local duplication). Low urgency given zero
  production exposure.

**Tier 3 — blocked on other in-flight batches, do not schedule yet:**
- A's leaves 3/4 (rewards-bearing) plumbing — blocked on batch (a) landing
  first; plumbing ahead of (a) would ship an inert kwarg.
- A's leaf 1 (daisy-chain) plumbing — blocked on batch (d) landing first;
  there is nothing to plumb into yet.

**Suggested full-batch framing, if a single "SVGD plumbing" batch is still
wanted as a deliverable:** ship Tier 1 now as its own small batch (it has
immediate, unconditional value and zero cross-batch coupling). Treat
"finish `Graph.svgd()` plumbing for leaves 1/3/4" as a *follow-up* task
explicitly gated on (d) and (a) respectively landing first, not as part of
the same batch — bundling them now would mean shipping dead code paths that
silently do nothing until unrelated, independently-scheduled work completes,
which conflicts with this project's no-silent-fallback principle if not
explicitly logged as such.

---

## Risk/unknowns list

1. **A, leaf 2b (exposure-set joint-index):** plumbing `exact_grad=True`
   through would make this sub-case reachable, but it has **no dedicated
   test coverage today** confirming the exact path is actually correct in
   combination with the exposure-scaling wrapper (`_wrap_model_with_exposure`,
   referenced but not read in depth in this investigation) — worth a
   dedicated de-risk pass before shipping, not assumed safe by transitivity
   from the non-exposure exact-path tests.
2. **B (baked-mode scatter-add):** the derivation was checked
   algebraically (gather/scatter-add VJP duality) but **not** verified
   numerically against a native FD/central-difference oracle in this
   investigation (no code was written or run — this was a scoping pass only,
   per the task's explicit "NOT implementation" framing). A future
   implementation batch must still do the same build-free de-risk +
   adversarial review rhythm every other B3 batch has gone through before
   landing, per this project's standing practice.
3. **B / D6 sequencing:** whether to do D6 first or bolt baked-mode support
   onto the current `lax.cond` wiring is presented as a genuine judgment
   call in B.4, not resolved here — matches this project's pattern of
   surfacing such trade-offs explicitly rather than deciding unilaterally.
4. **C.1:** the fix scope was fully confirmed by direct comparison against
   three working siblings; no residual unknown found. The one thing not
   checked: whether `moments_from_graph`'s FD backward (`moments_fn_bwd`)
   has any OTHER latent vmap issue beyond `_compute_moments_pure` — reasoned
   through (ordinary jnp ops are vmap-transparent) but not executed/tested
   in this investigation.
5. **C.2:** both guards are confirmed-missing and low-severity (zero
   production exposure, validator-only). The MPFR-gate wrinkle (touch shared
   helper vs. duplicate locally) is a real open decision, not a blocker.
6. **General:** this document is a scoping/feasibility pass only — no code
   was written, no tests were run, no existing behavior was changed. All
   "concrete required changes" sections above describe what a future
   implementation batch would need to do, not work already done.
