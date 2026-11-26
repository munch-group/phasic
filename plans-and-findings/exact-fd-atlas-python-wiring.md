# Exact-gradient vs finite-difference (FD) atlas — Python model-builder wiring

Every Python entry point in `phasic` that produces a JAX-differentiable
model, and exactly how (or whether) it uses a B3 exact theta-adjoint vs FD.
All line numbers verified by direct reading on 2026-08-04.

## Summary table

| Function | Quantity | Exact kwarg (default) | Wiring mechanism | Static exclusions | Dynamic exclusions | Rewards-safe? | fixed_mask-safe? |
|---|---|---|---|---|---|---|---|
| `Graph.pmf_from_graph` (`__init__.py:3524`) | PMF/PDF | **none** | none — 100% FD | n/a | n/a | n/a (no rewards arg) | n/a (no fixed_mask arg) |
| `Graph.pmf_from_graph_parameterized` (`__init__.py:3846`) | PMF/PDF | **none** | **disabled** — raises `NotImplementedError` | — | — | — | — |
| `Graph.pmf_from_cpp` (`__init__.py:4027`) | PMF/PDF | **none** | none — 100% FD | n/a | n/a | n/a | n/a |
| `Graph._daisy_chain_svgd_model` (`__init__.py:4254`, used by `svgd(epoch_starts=…)` / `epoch_model`) | joint stop-probs / sojourn per epoch chain | **none** | none — 100% FD (2 branches: no-exposure `_autodiff`, per-obs `_per_obs_autodiff`) | n/a | n/a | `rewards` param accepted, **completely inert** (not a "wrong value" bug — output is provably rewards-independent) | yes — `fd_skip_indices`/`fixed_set_local` skip FD in the one branch that exists |
| `Graph.method_of_moments` → `method_of_moments.py:205` | moments (via GMM/least-squares) | **none at the optimizer level** | scipy `least_squares` with no `jac=` → scipy's own 2-point FD, **independent of** the underlying model's `exact_moment_grad` (never invoked — `jax.grad` is never called) | — | — | pass-through only, not evaluated as a gradient | n/a |
| `Graph.moments_from_graph` (`__init__.py:6594`) | raw moments | **none** (confirmed: signature is `(cls, graph, nr_moments=2, use_ffi=False)`) | none — 100% FD | `weight_mode` must be `'linear'` or raises `ValueError` (not silent) | n/a | n/a (no rewards arg) | n/a |
| `Graph.pmf_and_moments_from_graph` (`__init__.py:6822`) | PMF/PDF + moments | `exact_moment_grad=True` (`__init__.py:6826`) | **`jnp.where`** (`__init__.py:7617`) — both FD and exact are computed whenever exact is enabled and rewards is None | `weight_mode ∉ {None,'linear','log'}` → FD; `weight_mode=='log'` AND effective-discrete → FD (`__init__.py:6972-6989`) | NaN-Jacobian sentinel from C → FD for that call (`__init__.py:7024-7033`, gated `__init__.py:7572`) | **Yes — guarded.** `rewards is not None` → exact forced off dynamically, logged (`__init__.py:7559-7566`); this is the fixed version of the previously-found bug (commit `315ce9c8`) | Yes — `_fixed_dims` skips the whole loop body incl. the exact term (`__init__.py:7584-7586`) |
| — its `cdf_zero_fn` / `cdf_zero_fn_cb` side-channels (`__init__.py:7193`, `7473`) | P(zero reward) for zero-inflation | **none** | none — 100% FD | n/a | n/a | n/a (rewards is an *input*, not optional) | Yes (`_fixed_dims`, `__init__.py:7208`, `7488`) |
| `Graph.pmf_from_graph_joint_index` (`__init__.py:7652`) | expected-sojourn / joint index probs | `exact_grad=False` (`__init__.py:7656`) | **`jax.lax.cond`** (`__init__.py:8268`) — but the "cheap" quotient-rule callback runs unconditionally either way; `cond` only gates the *expensive* 2·P FD loop | `weight_mode=='log'` → hard `ValueError` (not FD fallback, `__init__.py:7784-7792`); `weight_mode ∉ {None,'linear'}` (formula/callback) → FD; `was_dph` → FD; baked `observed_indices` dedup mode → FD; `theta_dim` overriding graph's own `param_length()` → FD (`__init__.py:7870-7916`) | Empty list from C → NaN sentinel → FD for that call (`__init__.py:7946-7954`, gated `__init__.py:8255`) | Not supported at all (declared, not silently wrong): docstring says "Ignored (must be None for joint_index mode)" (`__init__.py:7759`); `Graph.svgd` skips reward validation when `joint_index=True` (`__init__.py:5860`) | Yes — `fixed_indices_set` used in `_fd_theta_bar` (`__init__.py:8205-8206`) **and** multiplied into `exact_tbm` (`__init__.py:8261-8266`) |
| `Graph.pmf_and_moments_from_graph_multivariate` (`__init__.py:8283`) | multivariate PMF+moments | **none of its own** (signature confirmed: no `exact_moment_grad` param, `__init__.py:8283-8286`) | delegates to `pmf_and_moments_from_graph`'s `jnp.where`, **but see note below** | inherits `pmf_and_moments_from_graph`'s statics | inherits, **plus**: every call it makes passes a real (non-`None`) `rewards_j` (`__init__.py:8408-8420`, `8439-8448`), so the callee's dynamic reward guard fires on *every* per-feature call → exact grad is de facto **always off** for this function's actual (2D-rewards) use case | Safe (inherits the guard) but see note: exact path is effectively unreachable here | Forwarded via `fixed_mask=fixed_mask` to the underlying `model_1d` (`__init__.py:8377`) |
| `Graph.daisy_chain_joint_probs` (`__init__.py:10084`) | joint-probs after daisy chain (ad-hoc/one-shot use) | **none** | none — 100% FD | n/a | n/a | n/a (no rewards arg) | Yes — `fixed_set` (`__init__.py:10335-10338`, used at `10351-10353`) |
| `Graph.reward_visit_probability` (`__init__.py:3305`) → `ffi_wrappers.compute_reward_visit_probability_ffi` (`ffi_wrappers.py:1162`) → `_reward_visit_probability_autodiff` (`ffi_wrappers.py:1278`) | P(visit rewarded vertex before absorption) | **none** | none — 100% FD (`_rvp_bwd`, `ffi_wrappers.py:1294-1313`) | n/a | n/a | n/a — not a "rewards" argument in the SVGD sense; `target_vertices`/`initial_ipv` cross the callback boundary as genuine per-call args, `theta.at[i]` gradient only | n/a (no fixed_mask arg) |
| `bffg.py::bffg_log_prob(..., return_model=True)`'s `model` (`bffg.py:396`) | P(terminal state \| θ) via sojourn ratio | **none, and no FD either** | **no differentiation rule at all** — raw `jax.ffi.ffi_call` (`ffi_wrappers.py:1050-1059`), no `custom_vjp`/`custom_jvp` anywhere wraps it | n/a | n/a | n/a | n/a |

Note on the multivariate row: CLAUDE.md (lines 135-139) says the multivariate wrapper "inherits the default-on exact path... automatically." That is true only for the degenerate `rewards=None` call (`__init__.py:8385-8387`). For its actual purpose — 1D or 2D `rewards` per feature — every underlying `model_1d` call passes non-`None` `rewards`, and `pmf_and_moments_from_graph`'s *own* dynamic reward guard (the fix for the earlier bug) then forces FD on every single feature, every single call. So in practice `pmf_and_moments_from_graph_multivariate` is FD-only for moments in its real workload, despite technically wiring through the exact-capable function.

---

## 1. `Graph.pmf_from_graph` — `src/phasic/__init__.py:3524-3843`

Builds `(theta, times) -> pmf_values`. Three sub-paths depending on
`serialized['weight_mode']` / `graph.parameterized()`:

- **`weight_mode='callback'`** (`3636-3701`): Python weight callback +
  `phasic_pybind.parameterized.GraphBuilder.compute_pmf`, wrapped in
  `jax.pure_callback(..., vmap_method='sequential')` (`3664-3673`).
  `@jax.custom_vjp` at `3676`; `jax_model_bwd` (`3684-3696`) is
  **unconditional central-difference FD**, `eps=1e-7`, looped over every
  `theta` dim — no exact branch, no fixed_mask skip.
- **parameterized, non-callback (linear/log via FFI)** (`3703-3828`):
  `compute_pmf_ffi` (raw `jax.ffi.ffi_call`, `vmap_method="expand_dims"`,
  `ffi_wrappers.py:523-593`). `@jax.custom_vjp` at `3796`; `jax_model_bwd`
  (`3805-3823`) again **unconditional FD**.
- **non-parameterized**: delegates to `pmf_from_cpp` (§3).

There is **no `exact_*` kwarg anywhere in this function** — confirmed by
reading the full signature (`3525`) and both `jax_model_bwd`
implementations. This is notable: `pmf_from_graph` is the most commonly
documented "hello world" entry point in the module's own docstrings, and
it has never received the B3 treatment; only the *moments* half of
`pmf_and_moments_from_graph` did.

Rewards: no `rewards` parameter at all — N/A. `fixed_mask`: no parameter
at all — N/A.

## 2. `Graph.pmf_from_graph_parameterized` — `__init__.py:3846-3901` (+ dead code to `4218`)

Disabled: `raise NotImplementedError(...)` at `3896-3901` fires
unconditionally before any of the (unreachable) legacy C++-codegen
implementation below it runs. See CLAUDE.md "Disabled paths" — confirmed
directly: bug 5a (no `_ensure_jax_active()`), 5b (hardcoded `jnp.float32`
instead of F64), F-001 (discrete wrapper still calls `g.normalize()`,
`3991-3992` in the dead code) are all still present in the preserved-but-
unreachable body. No exact-grad question applies to dead code.

## 3. `Graph.pmf_from_cpp` — `__init__.py:4027-4216`

Loads a user C++ `build_model(theta, n_params)`, wraps
`compute_pmf`/`compute_dph_pmf` via `ctypes`. `_compute_pmf_pure`
(`4174-4183`) is a `jax.pure_callback(..., vmap_method='expand_dims')`
crossing `theta, times`. `@jax.custom_vjp` at `4186`; `jax_model_bwd`
(`4195-4213`) is **unconditional central-difference FD**, `eps=1e-7`. No
exact kwarg, no rewards, no fixed_mask. Note the load-bearing comment at
`4126-4135`: the discrete wrapper deliberately does **not** call
`g.normalize()` (the fix for "bug 4", the same class of defect flagged as
unfixed in the disabled `pmf_from_graph_parameterized` above).

## 4. `Graph._daisy_chain_svgd_model` — `__init__.py:4254-5056`

Internal builder behind `Graph.svgd(epoch_starts=...)` (`5157`) and
`Graph.epoch_model` (`5157` also, since `epoch_model` at `5057` calls the
same helper). Builds a JSP graph (`joint_stop_prob_graph`) and daisies
theta through epochs via `compute_daisy_chain_joint_probs_ffi` /
`compute_daisy_chain_sojourn_ffi` (raw `ffi_call`s, `vmap_method="expand_dims"`,
`ffi_wrappers.py:1411-1420` and `1423ff`).

Two branches, both **100% FD, no exact kwarg**:

- **No-exposure branch** (`4554-4714`): `_forward` wrapped in
  `jax.custom_batching.custom_vmap` (`4654-4666`, fuses per-particle SVGD
  vmap into one fat FFI call — a performance trick, not a gradient
  trick). `@jax.custom_vjp` at `4669`; `_autodiff_bwd` (`4676-4690`) is
  central-difference FD, `eps_local=1e-7`, skipping indices in
  `fixed_set_local` (built from `fd_skip_indices`, `4523-4526`, which in
  turn comes from `fixed_indices` only when `bake_fd_skip=True`, the
  `Graph.svgd` default — `epoch_model` passes `bake_fd_skip=False`,
  `4514-4522`).
- **Exposure branch** (`4715-4953`): same pattern, `@jax.custom_vjp` at
  `4900` (`_per_obs_autodiff`), FD backward at `4907-4930`.

**Rewards:** both `model(theta, _observed_arg=None, rewards=None)`
closures (`4694`, `4934`) accept `rewards` but **never reference it** in
the body (verified: the only occurrences of the string `rewards` in the
whole function's byte range are the two signatures and two docstring
mentions — `grep` confirms no other use). This is a different failure
mode than the earlier-fixed "silently wrong" bug: the output is
*provably independent of* `rewards` (it isn't reward-transformed at
all), so a caller passing `rewards=` here gets a silent no-op rather than
a silently-wrong value — still worth flagging, since nothing warns the
caller.

**fixed_mask:** handled consistently (only one gradient method exists,
so there's no exact/FD inconsistency to check) via `fd_skip_indices` in
both branches.

## 5. `Graph.method_of_moments` — `__init__.py:6393-6498` → `method_of_moments.py:205-...`

`Graph.method_of_moments` (`6393`) is a thin dispatcher (resolves
`theta_dim`, `discrete`, calls `_mom`). The real logic,
`method_of_moments.method_of_moments` (`method_of_moments.py:205`),
builds a model via `Graph.pmf_and_moments_from_graph_multivariate`
(`method_of_moments.py:389`, for 2D rewards) or
`Graph.pmf_and_moments_from_graph` (`401`, `413`, for 1D/no rewards) —
**with no `exact_moment_grad=` passed**, so those calls get whatever the
default is (`True`), but it doesn't matter: `moments_fn` (`396-421`)
immediately converts the JAX output to `np.asarray(...)`, and is then
handed straight to `scipy.optimize.least_squares(residual_fn, x0,
bounds=(1e-10, np.inf), method='trf', max_nfev=200*n_free)` (`473-478`,
and again at `489-494` for the weighted GMM refinement step) — **no
`jac=` kwarg anywhere**. scipy's `trf` method defaults to a `'2-point'`
finite-difference Jacobian when `jac` is omitted. `jax.grad` is never
invoked on the model here, so the model's own exact/FD gradient wiring
is entirely bypassed — this matches CLAUDE.md's claim exactly, confirmed
by direct reading, not by trusting the doc.

## 6. `Graph.moments_from_graph` — `__init__.py:6594-6819`

Confirmed **no `exact_moment_grad` kwarg**: full signature is
`moments_from_graph(cls, graph, nr_moments=2, use_ffi=False)` (`6594`) —
no third gradient-related parameter exists. Restricted to
`weight_mode='linear'` by a loud `ValueError` (`6683-6691`, with an
explicit comment documenting a real historical bug: a `'log'` graph
silently got wrong linear moments before this guard existed). Generates
one-off C++ (`_generate_cpp_from_graph`) and computes moments via
`g.expected_waiting_time(rewards)` chained `nr_moments` times
(`6712-6746`) through `ctypes`. `_compute_pure` (`6782-6786`) is a
`jax.pure_callback(..., vmap_method='expand_dims')` crossing `theta`
only. `@jax.custom_vjp` at `6789`; `moments_fn_bwd` (`6799-6816`) is
**unconditional central-difference FD**, no logging distinguishing
exact/FD (there is nothing to distinguish — this path never had B3
applied). No `rewards` parameter, no `fixed_mask` parameter — both N/A.

## 7. `Graph.pmf_and_moments_from_graph` — `__init__.py:6822-7649`

The primary B3 exact-gradient entry point.

**Kwarg:** `exact_moment_grad: bool = True` (`6826`).

**Quantity / weight_modes:** `model(theta, times, rewards=None) ->
(pmf_values, moments)`. Exact scope computed once, up front
(`6963-6989`):
```
_wm = serialized.get('weight_mode', 'linear')
_effective_discrete = bool(discrete) or bool(serialized.get('is_discrete', False))
_log_scope_ok = (_wm == 'log' and not _effective_discrete)
_linear_scope_ok = (_wm in (None, 'linear'))
```
`exact_moment_grad=False` (explicit) → FD, logged (`6974-6979`).
`weight_mode` outside `{None,'linear'}` and not (`'log'` AND
continuous) → FD, logged (`6980-6987`). So: `'formula'`/`'callback'` are
always excluded; `'log'` is excluded whenever the graph is effectively
discrete (native DPH or `was_dph`/`Graph.discretize()`), per the
load-bearing comment at `6966-6971` (confirmed by direct repro per the
comment, not just theoretical reasoning).

**Host callback / exact Jacobian** (`6990-7040`): private clone
`_exact_graph = graph.clone()` (`6992`). Inside `_one(t)`:
```python
_exact_graph.update_weights(t, log=_exact_is_log)   # 7012 — mirrors the model's own weight rule
if _effective_discrete:  J = _exact_graph._moments_grad_theta_dph(K, t.tolist())   # 7015 (phasic.c:11142)
elif _exact_is_log:      J = _exact_graph._moments_grad_theta_log(K, t.tolist())   # 7019 (phasic.c:10917)
else:                     J = _exact_graph._moments_grad_theta(K)                  # 7022 (phasic.c:10738)
```
Dynamic decline: if `J.size != nr_moments*param_length`, logs at INFO
(`7025-7031`) and returns a `NaN`-filled array (`7032-7033`) — the NaN
sentinel is the per-call fallback signal (not a raise, not silently
wrong — logged).

**Wiring mechanism: `jnp.where`, not `lax.cond`** (`7617`):
```python
grad_moments_i = jnp.where(_exact_ok, _exact_tbm[i], grad_moments_i)
```
`_exact_ok = jnp.all(jnp.isfinite(_exactJ))` and `_exact_tbm =
_exactJ.T @ g_moments` are computed **once, unconditionally**, whenever
`_exact_grad_enabled and not _rewards_provided` (Python-level static
condition) — i.e., this is *not* gated by a JAX-traced predicate the way
`pmf_from_graph_joint_index`'s is. Separately, the FD probes
(`pmf_plus/pmf_minus`, `moments_plus/moments_minus` via two
`_compute_pure` calls) run **unconditionally for every non-fixed
parameter regardless of whether the exact path is enabled**, because
they're also needed for the PMF gradient (which has no exact
counterpart at all — only moments got B3 treatment here). So in the
common case (`exact_moment_grad=True`, in-scope, no rewards), **both the
exact Jacobian (one pure_callback) and the full FD sweep (2·P forward
calls) are always computed**; `jnp.where` only decides which value to
*use* per output — this is exactly the surprising "both branches
computed unconditionally" pattern the task asked about, and it's the
opposite pattern from `pmf_from_graph_joint_index`'s `lax.cond`.

**Rewards — SAFE (the fixed bug):** `_rewards_provided = rewards is not
None and jnp.asarray(rewards).size > 0` (`7559`). If true, exact is
dynamically forced off for that call, logged (`7561-7566`) — this is the
fix for the previously-found "rewards silently ignored by the exact
Jacobian" defect (CLAUDE.md line 96, commit `315ce9c8`). Confirmed
correct today: the private clone `_exact_graph` (`6992`) is never
reward-transformed and neither `_moments_grad_theta`/`_dph`/`_log` take
a `rewards` argument, but the code now explicitly guards against ever
using that Jacobian when rewards are actually in play.

**fixed_mask:** `_fixed_dims = _fixed_indices_set_from_mask(fixed_mask)`
(`6937`). In the bwd loop, `if i in _fixed_dims: theta_bar.append(0.0);
continue` (`7584-7586`) — this `continue` happens *before* the
exact/FD combination code, so fixed dims never see either gradient
method; consistent across both.

**Companion `cdf_zero_fn` / `cdf_zero_fn_cb`** (zero-inflation support,
`7093-7231` for callback mode, `7339-7506` for pybind mode):
`@jax.custom_vjp` at `7193` and `7473` respectively — **both are
100% FD** (`cdf_zero_bwd_cb` at `7201-7223`, `cdf_zero_bwd` at
`7481-7504`), no exact counterpart exists for this quantity at all, not
even conditionally. Both correctly respect `_fixed_dims`.

## 8. `Graph.pmf_from_graph_joint_index` — `__init__.py:7652-8280`

**Kwarg:** `exact_grad: bool = False` (`7656`) — the only B3 kwarg in the
codebase that defaults to `False`; the docstring (`7716-7737`) and
CLAUDE.md (lines 169-197) both explain why (forward-mode, cost scales
with `P`; under `vmap`, `lax.cond` can't actually skip work).

**Weight-mode gate is a hard construction-time reject for `'log'`**
(`7783-7792`), not merely a fallback-to-FD, because the sojourn FFI
handler hardcodes linear weights (`ComputeSojournTimesFfiImpl`) — using
it on a `'log'` graph would be silently wrong, so it's rejected outright
regardless of `exact_grad`.

**Exact-scope statics** (`7868-7916`), each declining to FD with an
INFO log, checked in order: `exact_grad=False` explicit (`7871-7876`);
`weight_mode ∉ {None,'linear'}` — i.e. `'formula'`/`'callback'`
(`7877-7883`); `was_dph` (`Graph.discretize()`'d graph — native DPH,
`is_discrete=True`/`was_dph=False`, is **not** excluded, per
`7884-7891`); baked `observed_indices` dedup mode (`7892-7898`);
`theta_dim` overriding the graph's own `param_length()` (`7899-7914`,
added specifically because the C function reads `param_length` off the
clone, which would silently disagree with the serialized shape
otherwise).

**Host callback — theta AND vertex_indices, a genuine per-call
argument.** `_exact_sojourn_jac_np(theta_np, vertex_indices_np)`
(`7924-7961`) is explicitly documented in-line (`7930-7935`) as
different from the theta-only pattern used elsewhere: `vertex_indices`
is read from the model's runtime observed-data argument each call, not
closed over at construction, so the union of observed + all-terminal
indices must be computed **inside** the callback on concrete values. It
crosses the `pure_callback` boundary via `theta, _vi_norm` at
`8248-8254`, `vmap_method='sequential'`. This is flagged exactly as the
task asked: it is architecturally different from
`pmf_and_moments_from_graph`'s theta-only exact callback (`§7`), and it
is precisely *why* this function's exact path is excluded from `baked`
observed_indices mode — a construction-time dedup optimization can't be
applied to an index array that's a genuine runtime input without a
scatter-add through the inverse-index map (flagged as future work,
CLAUDE.md lines 210-213).

**Dynamic decline:** empty list `raw` from the C function → `NaN`
sentinel of the required shape for both `J_obs`/`J_all`, logged
(`7946-7954`).

**Wiring mechanism: `jax.lax.cond`, confirmed** (`8268`):
```python
theta_bar = jax.lax.cond(exact_ok, lambda: exact_tbm, _fd_theta_bar)
```
But nuance: the "cheap" part of the exact computation — two small FFI
sojourn calls (`8242-8246`) plus the host-callback Jacobian
(`8248-8254`) — runs **unconditionally before** the `cond`, per the
in-line rationale (`8221-8232`): "Always pay the CHEAP union-based
callback... only pay the EXPENSIVE 2·n_params FD loop... via lax.cond."
So `cond` here only gates the *FD loop itself* (`_fd_theta_bar`, a
closure, not pre-evaluated), which is the one part `lax.cond` can
actually skip when unbatched — consistent with the documented caveat
(and CLAUDE.md lines 178-183) that under `vmap(grad(loss))(particles)`
this skip does not materialize (XLA computes both `cond` branches when
the predicate is batched), making `exact_grad=True` a net *extra* cost
under SVGD's typical usage at this model's native `P`≈2.

**Rewards:** declared unsupported (`rewards: None — Ignored`, `7759`),
not silently misapplied; `Graph.svgd` skips reward validation entirely
when `joint_index=True` (`5860`).

**fixed_mask:** `fixed_indices_set` (`8186-8189`) used in both
`_fd_theta_bar` (`8205-8206`) and multiplied elementwise into
`exact_tbm` via `_fixed_keep` (`8261-8266`) before the `cond` —
consistent between the two branches.

## 9. `Graph.pmf_and_moments_from_graph_multivariate` — `__init__.py:8283-8477`

No `exact_moment_grad` parameter in the signature (`8283-8286`,
confirmed by reading, not assuming). Builds `model_1d =
cls.pmf_and_moments_from_graph(...)` **without** forwarding any exact-
grad kwarg (`8375-8378`), so `model_1d` always gets the function's
default (`True`). It then loops per-feature (`8406-8422` sparse,
`8437-8452` dense) calling `model_1d(theta, times_j, rewards=reward_j)`
with a real, non-`None` reward vector every time. Per §7's dynamic
guard, this means `pmf_and_moments_from_graph`'s exact path is
dynamically disabled on **every single one of these calls** — so despite
CLAUDE.md's characterization ("inherits the default-on exact path... and
its logging automatically", lines 135-139), in its actual 1D-or-2D-
rewards use case this function is **FD-only for the moments gradient in
practice**, not merely "inheriting a default." Only the no-op
`rewards=None` branch (`8385-8387`) would actually reach the exact path.
`fixed_mask` is forwarded correctly (`8377`).

## 10. `Graph.daisy_chain_joint_probs` — `__init__.py:10084-10363`

Public, ad-hoc/one-shot sibling of `_daisy_chain_svgd_model` (§4) — used
directly rather than built once and reused (note at `10298-10311`
explicitly says not to add `custom_vmap` here because the closures are
re-created every call, which would leak tracers under
`vmap(jit(grad(...)))`). `weight_mode=='log'` is a hard `ValueError`
(`10173-10180`), same FFI-hardcodes-linear rationale as §8. `_forward`
(`10312-10324`) calls the same raw daisy-chain FFI functions.
`@jax.custom_vjp` at `10340`; `_autodiff_bwd` (`10347-10359`) is
**100% central-difference FD**, no exact kwarg exists. No `rewards`
parameter at all (N/A). `fixed_indices` (`10091`) → `fixed_set`
(`10335-10338`) skips FD consistently (`10351-10353`) — only one
gradient method exists here, so there is no cross-branch consistency
question, just internal consistency, which holds.

## 11. `Graph.reward_visit_probability` — `__init__.py:3305-3396`

Architecturally different from the others: it is a plain instance
method that computes-and-returns a value directly (not a
"build-once-call-many" model factory returning a `model` closure). For
JAX-traced `theta` it routes through
`ffi_wrappers.compute_reward_visit_probability_ffi` (`ffi_wrappers.py:
1162-1206`) → `_reward_visit_probability_autodiff`
(`ffi_wrappers.py:1278-1284`, `@functools.partial(jax.custom_vjp,
nondiff_argnums=(0,))`). Forward: `_reward_visit_probability_forward`
(`1253-1275`) is a `jax.pure_callback(..., vmap_method='sequential')`
crossing `theta, target_vertices, initial_ipv` — all three cross the
boundary because `target_vertices` (derived from the caller's `rewards`
argument's nonzero pattern, `__init__.py:3357`) and `initial_ipv` are
genuine per-call values for this API shape (there is no persistent
"model" object to close over them in). Backward: `_rvp_bwd`
(`ffi_wrappers.py:1294-1313`) is **unconditional central-difference FD**
over `theta` only (`target_vertices`/`initial_ipv` get zero cotangents,
`1310-1312`). **No `exact_*` kwarg exists anywhere in this call chain**
— this had been an initial hypothesis-to-check (an FFI-native quantity
might get analytic gradients for free), and it's false: confirmed by
grep that no `custom_vjp`/`custom_jvp` other than the one at
`ffi_wrappers.py:1278` touches this function, and that one is FD.

## 12. `bffg.py::bffg_log_prob(..., return_model=True)` — `bffg.py:253-611`

`model(theta_mcmc, vertex_indices, rewards=None)` (`bffg.py:396-427`)
calls `compute_sojourn_times_ffi` (`ffi_wrappers.py:936-1060`) **directly**
— that function is a raw `jax.ffi.ffi_call("ptd_compute_sojourn_times",
..., vmap_method="expand_dims")` (`ffi_wrappers.py:1050-1059`) with **no**
Python-level `custom_vjp`/`custom_jvp` wrapper anywhere in
`ffi_wrappers.py` (confirmed by grep — the only `custom_vjp` in that
file wraps a *different* function, `_reward_visit_probability_autodiff`,
§11) and no C-side registered differentiation rule found in this
codebase (`primitive_jvps`/`primitive_transposes`/etc. — none present).
Consequence: `jax.grad` applied directly to this `model` (or to
`likelihood_correction_jit`, `bffg.py:546-563`, which also calls
`sample_path_conditioned_ffi`, another undifferentiated raw `ffi_call`)
would raise, not silently return an FD or wrong-but-plausible gradient —
this is a **third distinct failure mode** beyond "exact" and "FD": no
differentiation rule exists at all. In practice this is inert: the sole
consumer, `MCMC` (`mcmc.py`), never calls `grad`/`value_and_grad`
anywhere (confirmed by grep — zero matches for `grad` in `mcmc.py`); it
is a value-only sampler (Metropolis-Hastings style) that only evaluates
`model`/`likelihood_correction` at concrete θ. So `bffg.py` has **no
exact-grad kwarg, and does not use FD either** — gradient support for
this path is simply unimplemented, not merely defaulted off.

---

## Headline findings

1. **`pmf_and_moments_from_graph` (`__init__.py:6822`, the main B3 function) uses `jnp.where`, and both branches genuinely run unconditionally** — the exact Jacobian (one `pure_callback`) and the full 2·P-call FD sweep both execute every time `exact_moment_grad=True` and no rewards are given, because FD is still needed for the PMF half regardless (`__init__.py:7568-7617`). This is the opposite of `pmf_from_graph_joint_index`, which uses `lax.cond` (`__init__.py:8268`) specifically to *avoid* that cost — though even there, only the FD loop is skipped; the cheap exact-side FFI calls always run.

2. **The rewards bug fixed previously in `pmf_and_moments_from_graph` is properly guarded**: rewards now dynamically force FD, logged (`__init__.py:7559-7566`). But **`pmf_and_moments_from_graph_multivariate` inherits this guard in a way that makes exact grad practically dead** — every per-feature call passes real rewards, so the guard fires every time, contradicting CLAUDE.md's "inherits the default-on exact path" framing in practice (not in code correctness, just in reachability).

3. **`bffg.py`'s `model`** calls a raw FFI primitive with *zero* registered differentiation rule anywhere — not FD, not exact, just undifferentiable. Harmless only because `mcmc.py` never differentiates it.

4. **Half the entry points have no exact kwarg at all and never did**: `pmf_from_graph`, `pmf_from_cpp`, `moments_from_graph`, `daisy_chain_joint_probs`, `_daisy_chain_svgd_model`, `reward_visit_probability` — all 100% FD, confirmed by reading each `*_bwd`.

5. `_daisy_chain_svgd_model` accepts a `rewards` kwarg it never references — a silent no-op, not a silently-wrong value.
