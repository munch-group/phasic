# B3 exact-gradient reachability from `Graph.svgd()`

**Purpose.** On 2026-08-01/04 a redesign was nearly built on the premise that
fixing a JAX control-flow issue in `pmf_from_graph_joint_index`'s
`exact_grad=True` wiring would speed up real SVGD runs. It turned out
`Graph.svgd()` never passes `exact_grad` through to that function at all, and
the common SVGD call pattern (`observed_indices` baked/dedup) is a *static*
exclusion for that function's exact path regardless of the kwarg — so the
entire premise of the fix was moot for real usage. This document traces
**every** dispatch branch inside `Graph.svgd()` (defined
`src/phasic/__init__.py:5241`), for **every** B3 `exact_*` kwarg
(`exact_moment_grad` on `pmf_and_moments_from_graph`, `exact_grad` on
`pmf_from_graph_joint_index`), so that question never again has to be
answered by chance.

All line numbers are `src/phasic/__init__.py` unless a file is given
explicitly (`src/phasic/svgd.py`). Every claim below was read from source in
this session, not inferred.

**Headline finding, upfront:** of five leaf branches in `Graph.svgd()`'s
dispatch tree, exactly **one** (no-rewards, non-joint-prob-graph) can reach a
B3 exact-gradient path today, and only because that leaf's callee happens to
default the kwarg to `True` — `Graph.svgd()` itself never sets or exposes
*any* `exact_*` kwarg anywhere in its ~40-parameter signature
(`:5241`–`:5281`, verified by reading the full parameter list; there is no
`exact_grad`, `exact_moment_grad`, `use_exact_gradient`, or similarly-named
parameter). The other four leaves are excluded, three of them
**unconditionally on every single gradient step of every run that takes that
branch** — not intermittently, not depending on theta.

---

## 1. Full dispatch tree

```
Graph.svgd(observed_data, ..., epoch_starts=None, rewards=None, exposure=None, joint_index=None, ...)
│
│  (Graph.svgd's signature, :5241-5281, has NO exact_* kwarg of any kind.)
│
├─ if self._joint_prob_base_graph_indexer is not None:        (:5940)   "joint-prob graph" — built via graph.joint_prob_graph(...)
│    joint_index forced True (:5953); discrete forced True (:5990)
│    │
│    ├─ if epoch_starts is not None:                          (:6017)   "daisy-chain" branch
│    │     → self._daisy_chain_svgd_model(...)                (:6045-6058)
│    │         → returns a model built by inline custom_vjp machinery
│    │           (compute_daisy_chain_joint_probs_ffi /
│    │            compute_daisy_chain_sojourn_ffi, via
│    │            Graph.daisy_chain_joint_probs :10084)
│    │         LEAF A — FD ONLY. No exact_* kwarg exists anywhere on this
│    │         path (verified: `_daisy_chain_svgd_model` signature :4254-4269
│    │         has no exact_* param; `daisy_chain_joint_probs` signature
│    │         :10084-10093 has no exact_* param either). The gradient is a
│    │         hardwired `@jax.custom_vjp` finite-difference backward
│    │         (":FD backward, skipping fixed indices" :4668-4669; "custom_vjp
│    │         wraps _per_obs_core to provide the FD backward" :4895-4900).
│    │
│    └─ else:  (no epoch_starts)                               (:6071)
│          _bake_obs = observed_data if exposure is None else None   (:6092)
│          → Graph.pmf_from_graph_joint_index(self, theta_dim=theta_dim,
│                fixed_mask=fixed_mask_for_model,
│                observed_indices=_bake_obs)                    (:6093-6097)
│          LEAF B — exact_grad kwarg EXISTS on the callee
│          (`exact_grad: bool = False`, :7656) but Graph.svgd never passes it
│          → callee's own default (False) always applies.
│            ├─ sub-case exposure is None (the common/default case):
│            │     _bake_obs = observed_data → baked/dedup mode is ON
│            │     → STATICALLY EXCLUDED regardless of exact_grad's value:
│            │       `elif _baked: ... _jix_exact_enabled = False`
│            │       (:7892-7898). Even if Graph.svgd were fixed to pass
│            │       exact_grad=True, this sub-case could never use it
│            │       without ALSO changing the baked-mode exclusion.
│            └─ sub-case exposure is not None:
│                  _bake_obs = None → baked mode is OFF, so the :7892
│                  exclusion does NOT fire. Structurally reachable in
│                  principle (subject to weight_mode='linear'/None and
│                  not was_dph, both graph properties Graph.svgd does not
│                  control) IF Graph.svgd plumbed exact_grad through —
│                  it currently does not, so it is unreachable in practice
│                  today, but not for a *structural* reason on this
│                  sub-branch.
│
├─ elif rewards is not None:                                   (:6099)   (only reached when NOT a joint-prob graph)
│    rewards_arr = jnp.asarray(rewards)
│    │
│    ├─ if rewards_arr.ndim == 2:                               (:6102)
│    │     → Graph.pmf_and_moments_from_graph_multivariate(
│    │           self, nr_moments=nr_moments, discrete=discrete,
│    │           use_ffi=False, theta_dim=theta_dim,
│    │           fixed_mask=fixed_mask_for_model)               (:6104-6108)
│    │     LEAF C — this function's signature (:8283-8286) HAS NO
│    │     exact_moment_grad PARAMETER AT ALL (cannot be set even if
│    │     Graph.svgd wanted to). Internally it builds
│    │     `model_1d = cls.pmf_and_moments_from_graph(...)` WITHOUT
│    │     passing exact_moment_grad (:8375-8378) → model_1d's own default
│    │     (True) applies at construction. BUT every per-feature call is
│    │     `model_1d(theta, times_j, rewards=reward_j)` with reward_j always
│    │     a concrete, non-None slice of the 2D rewards matrix (:8420 sparse
│    │     path, :8448 dense path) → DYNAMICALLY/STRUCTURALLY
│    │     EXCLUDED on every call, every iteration (see rewards exclusion
│    │     below).
│    │
│    └─ else:  (1D rewards)                                     (:6109)
│          → Graph.pmf_and_moments_from_graph(
│                self, nr_moments=nr_moments, discrete=discrete,
│                theta_dim=theta_dim, fixed_mask=fixed_mask_for_model)
│                                                                 (:6111-6114)
│          LEAF D — exact_moment_grad kwarg EXISTS (default True, :6826),
│          not overridden by Graph.svgd (default silently applies = True
│          AT CONSTRUCTION). BUT SVGD fixes `self.rewards = rewards`
│          (svgd.py:5702) once at construction and passes it unchanged into
│          EVERY forward/backward call via
│          `partial(self._log_prob_unified, ..., rewards=self.rewards)`
│          (svgd.py:6431) → `self.model(theta, self.observed_data,
│          rewards=rewards)` (svgd.py:6207). Since rewards is user-supplied
│          and non-None for this whole branch by construction, EVERY
│          gradient step hits the rewards check inside
│          `pmf_and_moments_from_graph`'s custom_vjp backward:
│          `_rewards_provided = rewards is not None and ...` (:7559) →
│          `if _exact_grad_enabled and _rewards_provided:` logs and falls
│          back to FD (:7561-7566). DYNAMICALLY/STRUCTURALLY EXCLUDED on
│          every call.
│
└─ else:  (rewards is None, not a joint-prob graph)              (:6115)
      → Graph.pmf_and_moments_from_graph(
            self, nr_moments=nr_moments, discrete=discrete,
            theta_dim=theta_dim, fixed_mask=fixed_mask_for_model)
                                                                   (:6117-6120)
      LEAF E — exact_moment_grad kwarg EXISTS (default True, :6826), not
      overridden by Graph.svgd, default silently applies = True. This
      branch always calls the model with rewards=None (self.rewards is
      None; svgd.py:5702, 6431), so the :7559-7566 rewards exclusion never
      fires. NOT STATICALLY EXCLUDED BY GRAPH.SVGD'S DISPATCH — the exact
      path CAN engage, subject only to the callee's own graph-dependent
      scope gate (weight_mode in {None,'linear'} always OK; weight_mode
      'log' OK only if the graph is continuous, :6972; per-theta MPFR
      declines still possible, :7024-7033) — properties of the *graph*
      the caller supplied, not of Graph.svgd's own call pattern.
      *** THIS IS THE ONLY LEAF REACHABLE TODAY WITH NO Graph.svgd CHANGE. ***
```

### Escape hatch: `SVGD(model=..., ...)` direct

`SVGD.__init__` (`src/phasic/svgd.py:4948-4975`) takes an opaque `model`
callable as its first argument and otherwise has **no** `exact_*` kwarg
either (its full parameter list, `:4948`-`:4974`, contains no
`exact_grad`/`exact_moment_grad`/similar). It does not construct a model
itself — it only consumes one.

This means a caller can build the model **directly**, with the exact kwarg
set explicitly:

```python
model = Graph.pmf_and_moments_from_graph(graph, exact_moment_grad=True)     # or
model = Graph.pmf_from_graph_joint_index(graph, exact_grad=True,
                                          observed_indices=None)             # must NOT bake
svgd = SVGD(model=model, observed_data=..., theta_dim=..., ...)              # bypasses Graph.svgd entirely
```

This is documented as the intended route for exactly this situation
(`src/phasic/svgd.py:4581-4589`: "Use `SVGD(model=..., ...)` direct only
when you have a pre-built JAX-compatible model callable ... Parameters
unique to `Graph.svgd` — `discrete`, `joint_index`, `tied`, `callback`,
`epoch_starts`, `daisy_chain_t_eval`, `daisy_chain_granularity`,
`daisy_chain_probe_theta`, `daisy_chain_t_eval_tol`, `validate_rewards`,
`return_history` — are pre-model-construction concerns and have no meaning
when SVGD is handed an opaque model callable"). Note this list of
"unavailable via the escape hatch" features itself excludes `rewards`,
`epoch_starts`/daisy-chain convenience, and `tied` from being usable
together with a hand-built exact-gradient model — a caller using this route
for `pmf_from_graph_joint_index` gives up `tied`; for daisy-chain there is
no exact-gradient model to build at all (Leaf A has none, escape hatch or
not).

---

## 2. Reachability table

| SVGD dispatch branch | leaf model-builder function | exact-kwarg passed by `Graph.svgd`? | structurally excluded regardless of kwarg? | reachable in practice today (no `Graph.svgd` change)? |
|---|---|---|---|---|
| joint-prob graph + `epoch_starts` set (daisy-chain) | `_daisy_chain_svgd_model` → `daisy_chain_joint_probs` (`:6045`, `:10084`) | N/A — **no `exact_*` param exists on either function** (`:4254-4269`, `:10084-10093`) | Yes — FD is the only implementation (`@jax.custom_vjp` FD backward, `:4668-4669`, `:4895-4900`); no exact variant to select | **No** |
| joint-prob graph, no `epoch_starts`, `exposure=None` (default/common) | `pmf_from_graph_joint_index` (`:6093`) | No — kwarg `exact_grad` exists (default `False`, `:7656`) but is never passed; default silently applies | Yes — `observed_indices` is baked whenever `exposure is None` (`:6092`), and baked mode is a static exclusion (`:7892-7898`) independent of `exact_grad`'s value | **No** |
| joint-prob graph, no `epoch_starts`, `exposure` set | `pmf_from_graph_joint_index` (`:6093`, with `_bake_obs=None`) | No — same as above, never passed | No — baked-mode exclusion does not fire here (`_baked=False`); only graph-dependent scope gates (weight_mode, was_dph) could exclude it | **No** (kwarg never plumbed, so callee default `False` always wins) — but *not* structurally blocked; would become reachable if `Graph.svgd` passed `exact_grad=True` here |
| non-joint-prob graph, 2D `rewards` | `pmf_and_moments_from_graph_multivariate` (`:6104`) → internally `pmf_and_moments_from_graph` (`:8375`) | No kwarg exists on the outer function at all (`:8283-8286`); inner call passes none either, so inner default (`True`) applies at construction | Yes, dynamically on every call — every per-feature call supplies a concrete non-None `reward_j` (`:8420` sparse, `:8448` dense), which the inner model's custom_vjp backward detects (`:7559`) and always falls back to FD (`:7561-7566`) | **No** |
| non-joint-prob graph, 1D `rewards` | `pmf_and_moments_from_graph` (`:6111`) | No — kwarg `exact_moment_grad` exists (default `True`, `:6826`); not passed, so default `True` silently applies at construction | Yes, dynamically on every call — `SVGD.rewards` is fixed non-None for the whole run (svgd.py:5702) and passed on every gradient step (svgd.py:6431, 6207) → rewards-check fallback (`:7559-7566`) fires every time | **No** |
| non-joint-prob graph, no `rewards` | `pmf_and_moments_from_graph` (`:6117`) | No — kwarg exists (default `True`, `:6826`); not passed, default `True` silently applies | No — `rewards` is always `None` on this branch (svgd.py:5702/6431/6207), so the rewards exclusion never fires; only graph-dependent scope (weight_mode, MPFR per-theta decline) can exclude it | **Yes** — the only leaf reachable with zero `Graph.svgd` changes, contingent on the caller's graph having `weight_mode` in `{None,'linear'}` or (`'log'` + continuous) |
| any branch, via `SVGD(model=..., ...)` direct | whichever model the caller builds | Caller sets it explicitly at model-construction time | Same per-function exclusions as above still apply (e.g. still can't force exact on daisy-chain — no such variant exists; still must avoid baked `observed_indices` for joint-index) | **Yes for `pmf_and_moments_from_graph`/`pmf_from_graph_joint_index` (unbaked)** — this is the one deliberate escape hatch (svgd.py:4948, docstring `:4581-4589`) |

---

## 3. Plain-English summary

`Graph.svgd()` has **no top-level knob at all** for choosing exact vs.
finite-difference gradients — not `exact_grad`, not `exact_moment_grad`,
nothing (`src/phasic/__init__.py:5241-5281`). Whatever gradient method ends
up running is purely a byproduct of (a) which of five leaf branches the
call falls into, and (b) whether that leaf's default for its own hidden
`exact_*` kwarg happens to be `True` and happens to not be dynamically
disqualified.

Walking the whole tree: the **daisy-chain path** (`epoch_starts` set) has no
exact-gradient implementation of any kind to plumb through — it is FD by
construction, full stop. The **joint-index path** (the common case for this
repo's population-genetics workloads, built via `graph.joint_prob_graph`) —
which is exactly the path the earlier redesign mistakenly targeted — is
never given `exact_grad` by `Graph.svgd`, and in its default, no-`exposure`
form is *also* statically excluded by baked/dedup mode regardless; only the
less-common `exposure`-set sub-case escapes that specific static exclusion,
and even then only because the kwarg still defaults to `False` unless
someone edits `Graph.svgd`. Both **rewards-bearing paths** (1D and 2D) call
into a function whose exact-gradient default is `True`, but SVGD fixes
`rewards` for the entire run and passes it on every single gradient step, so
the callee's own "rewards not supported by the exact path" guard trips
every time — this is a *new* finding from this trace, not previously
documented as a reachability gap, and it means two of the five leaves are
excluded for a reason that has nothing to do with `exact_grad`/
`exact_moment_grad` plumbing at all.

That leaves exactly **one** leaf — non-joint-prob graph, no `epoch_starts`,
no `rewards` — where the B3 exact-moment-gradient path can actually run
today from a plain `Graph.svgd(...)` call, and it does so silently, only
because `pmf_and_moments_from_graph`'s own default is `True` and nothing in
that branch disqualifies it. Every other leaf would need `Graph.svgd` itself
modified — to add a top-level kwarg and thread it through, and in the
baked-joint-index and both rewards cases, to *also* change something deeper
than plumbing (the baking policy, or the FD-on-rewards limitation) — before
it could ever reach an exact-gradient path. The one presently-working
escape hatch for a caller who wants exact gradients on a rewards or
joint-index model today is to bypass `Graph.svgd` altogether: build the
model directly via `Graph.pmf_and_moments_from_graph(..., exact_moment_grad=True)`
or `Graph.pmf_from_graph_joint_index(..., exact_grad=True, observed_indices=None)`
and hand it to `SVGD(model=..., ...)` directly — at the cost of losing
`Graph.svgd`'s convenience features (`tied`, daisy-chain wiring, reward
validation/zero-inflation attachment, etc., per `src/phasic/svgd.py:4581-4589`).
