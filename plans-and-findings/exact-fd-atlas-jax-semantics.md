# JAX control-flow semantics atlas for `phasic`

**Purpose.** This session discovered — only after writing code around the
opposite assumption — that `jax.lax.cond` computes BOTH branches when its
predicate is a batched array under `vmap`, silently defeating a
conditional-cost design. This document catalogs every JAX control-flow
pattern this codebase actually uses (`jax.lax.cond`, `jax.custom_vjp`,
`jax.pure_callback` + `vmap_method`, `jax.experimental.custom_batching.
custom_vmap`, `jax.vmap` itself, plus two closely-related patterns found
along the way: `jax.debug.callback` and `jax.lax.map`), states the precise
semantic rule for each, says exactly how the rule was verified, and cites a
real call site in this codebase that relies on it — correctly, or (where one
was found) incorrectly.

**Provenance.** All line numbers are `src/phasic/__init__.py` unless a file
is given explicitly (`src/phasic/svgd.py`). Every codebase claim below was
read from source in this session, not inferred. Every JAX-semantic claim was
either (a) already confirmed by `experiments/dr_lax_cond_vmap_derisk.py`
(cited as "D0" below), (b) newly confirmed by
`experiments/dr_jax_semantics_atlas_derisk.py` written for this document
(cited as "D1", with the specific claim number, e.g. "D1#8"), or (c) — for
one item where execution wasn't practical — clearly marked as JAX's
documented behavior rather than an empirical result of this session. JAX
version: `0.7.2` (`pixi run python -c "import jax; print(jax.__version__)"`).
Run either script to reproduce every number quoted here:

```bash
pixi run python experiments/dr_lax_cond_vmap_derisk.py
pixi run python experiments/dr_jax_semantics_atlas_derisk.py
# claim 13 (moments_from_graph vmap-break) needs source on disk:
PHASIC_SOURCE_DIR=/Users/kmt/phasic pixi run python experiments/dr_jax_semantics_atlas_derisk.py
```

---

## 1. `jax.lax.cond`: trace-time construction vs runtime execution

### 1a. The core rule

`lax.cond(pred, true_fn, false_fn, x)` always **traces** (Python-calls) both
`true_fn` and `false_fn` once, to build two jaxprs — this happens
unconditionally, regardless of `pred`, and is not itself evidence of runtime
cost. What differs is what XLA does with those two jaxprs **at runtime**:

- **`pred` is an unbatched (scalar) value** — under `jax.jit` alone, under
  `jax.grad` alone, or eagerly — `cond` lowers to XLA's `Conditional` HLO op,
  which performs genuine data-dependent branching: only the taken branch's
  computation actually runs on the device. The untaken branch costs
  (approximately) nothing.
- **`pred` is a batched value under `jax.vmap`** — because different batch
  elements may want different branches, XLA cannot use `Conditional`; it
  lowers to `select_n` (compute both operands, then select elementwise per
  batch element). **Both branches execute in full, for every call, at a
  cost that does not depend on what fraction of the batch actually wants
  each branch** — even a batch where the predicate is `True` for every
  single element still pays for the `False` branch too.

### 1b. Verification

- D0 (`experiments/dr_lax_cond_vmap_derisk.py`, claim 1): a spy-counter
  proof that under `vmap` with an always-`True` predicate, the `False`
  branch's Python body still gets invoked (trace-time evidence).
- D1#4/5 (`dr_jax_semantics_atlas_derisk.py`,
  `claim4_5_cond_runtime_skip_jit_vs_vmap`): the trace-time-only evidence
  above can't distinguish "traced once" from "executes every call", so this
  adds a **wall-clock timing** proof with a genuinely expensive branch
  (900×900 matmul) vs a trivial one:

  | composition | predicate | measured cost |
  |---|---|---|
  | `jit(cond)` | scalar `False` | 0.04 ms/call |
  | `jit(cond)` | scalar `True` | ~11–13 ms/call (**264–346×** the `False` case — genuine runtime skip) |
  | `jit(vmap(cond))`, 8 elems | all `False` | ~110–112 ms/call |
  | `jit(vmap(cond))`, 8 elems | all `True` | ~110–112 ms/call (**0.89–0.98×** — cost is predicate-*independent*) |

  The all-`False` vmap batch costs **~8.5–9.1×** one scalar expensive call —
  i.e. it is secretly paying for the expensive branch on every one of its 8
  elements, despite never wanting it.
- D1#5: `jax.grad` alone (no vmap) on a scalar-predicate `cond` selects the
  correct branch's derivative (`d(2x)/dx=2` at `x=+1`, `d(3x)/dx=3` at
  `x=-1`) — grad-of-cond composes normally when the predicate isn't batched.

### 1c. Nesting / sequential conds

D1#6 (`claim6_nested_cond_under_vmap`): the both-branches-under-vmap rule
applies **independently at every `cond` site**, including a `cond` nested
inside another `cond`'s branch. Nesting neither compounds the cost
(multiplicatively) nor shortcuts it (an outer branch being "selected" does
not let an inner `cond` inside it skip) — outer AND inner conds each traced
and computed both their own branches.

### 1d. A branch containing its own `custom_vjp`

D1#7 (`claim7_cond_branch_with_custom_vjp_under_vmap_grad`): when one of
`cond`'s branches is a function wrapped in `jax.custom_vjp`, under
`vmap(jax.grad(cond(...)))` **both the custom_vjp's `fwd` AND `bwd` get
traced** for that branch, even when the predicate is `True` for every
element in the batch (so the *other*, plain branch "shouldn't" need to run
at all, and yet does). The both-branches rule reaches through a
`custom_vjp` boundary in exactly the same way it reaches through a plain
function.

### 1e. Real codebase example — used correctly, with the limitation documented inline

`pmf_from_graph_joint_index`'s `model_bwd`, `__init__.py:8217–8268`:

```python
if _exact_sojourn_jac_np is None:
    theta_bar = _fd_theta_bar()
else:
    ...
    exact_ok = jnp.all(jnp.isfinite(J_obs)) & jnp.all(jnp.isfinite(J_all))
    ...
    theta_bar = jax.lax.cond(exact_ok, lambda: exact_tbm, _fd_theta_bar)
```

The surrounding comment (`:8224–8232`) states the exact tradeoff from §1a
explicitly and by name, arrived at via adversarial review:

> "Always pay the CHEAP union-based callback + two small FFI sojourn calls
> to check applicability for this theta; only pay the EXPENSIVE 2×n_params
> FD loop (via `lax.cond`, not `jnp.where`) when the exact path actually
> declines ... lax.cond's skip only manifests for a single, non-vmapped
> call; under SVGD's actual `vmap(grad(loss))(particles)` a batched
> predicate lowers to a select over both branches, same cost as
> `jnp.where` ... so `lax.cond` is never worse and sometimes better."

This is the correct call: `lax.cond` never costs *more* than `jnp.where`
would have (both compute both branches under vmap; only `lax.cond` has a
chance of skipping outside vmap), so there is no reason to prefer
`jnp.where` here even knowing vmap defeats the skip. Note this exact
function's **public** `exact_grad` kwarg still defaults to `False`
(`:7701–7737`) specifically *because of* this limitation compounding with
forward-mode cost scaling with P — see that docstring for the full
reasoning; the `lax.cond` choice inside `model_bwd` is a separate, smaller
decision from the kwarg default.

### 1f. Real codebase contrast — a sibling function chose `jnp.where` instead, for a different reason

`pmf_and_moments_from_graph`'s `model_bwd`, `__init__.py:7534–7622`, is the
**other** B3 exact-gradient site (`exact_moment_grad`, default `True`) and
makes the opposite choice — `jnp.where`, not `lax.cond` — to combine exact
vs FD:

```python
grad_moments_i = jnp.where(_exact_ok, _exact_tbm[i], grad_moments_i)
```

This is *also* correct, and the difference is instructive: in this
function the FD loop (`theta_plus`/`theta_minus` perturbations, `:7580
–7612`) has to run **regardless** of whether the exact path succeeds,
because it is the *only* source of the PMF gradient (the exact path covers
moments only, `:6939–6946`). Since the "cheap-looking" alternative was
never actually avoidable here, there is nothing for `lax.cond` to save —
`jnp.where`'s simplicity is preferred with no downside. `lax.cond` is worth
reaching for in `pmf_from_graph_joint_index` (§1e) specifically *because*
its FD branch (`_fd_theta_bar`, an entire independent 2×n_params forward-pass
loop) is otherwise skippable in full.

**Takeaway:** choosing `lax.cond` over `jnp.where` is worth it only when the
"cheap" branch is a cost you can *actually avoid* outside vmap; if both
branches' prerequisite work happens unconditionally anyway, `jnp.where` is
simpler and no more expensive under any composition.

---

## 2. `jax.pure_callback` exception propagation

### 2a. The rule

A Python exception raised inside a `pure_callback`'s host function
propagates to the caller as a genuine, catchable exception — in every
composition tested: bare, inside a `jax.custom_vjp`'s `fwd`, inside its
`bwd`, under `jax.jit(jax.grad(...))`, and under
`jax.vmap(jax.jit(jax.grad(...)))` — **the exact composition SVGD uses**
(see §2d). It is never silently swallowed and never turned into an opaque,
unrelated JAX-internal error.

**The exact type reaching the caller, however, changes once `jit` is in the
composition**: under `jit` (with or without `vmap`/`grad` wrapping it), the
callback error is re-raised as `jaxlib._jax.XlaRuntimeError`, **not** the
original `RuntimeError` object. Two things make this safe to treat as
"basically the same exception" for most purposes:

1. `jaxlib._jax.XlaRuntimeError` **is a genuine Python subclass of the
   builtin `RuntimeError`** (confirmed via `.__mro__` —
   `(XlaRuntimeError, RuntimeError, Exception, BaseException, object)`), so
   `except RuntimeError:` and `pytest.raises(RuntimeError)` both catch it
   without any special-casing.
2. The original message text is embedded verbatim inside the wrapper's
   `str(e)` (as part of an `"INTERNAL: CpuCallback error calling callback:
   ..."` traceback dump), so `pytest.raises(RuntimeError, match="...")`
   against the *original* message substring still matches (substring/regex
   search, not exact-string equality).

What does **not** survive: if the host function raises a **more specific**
exception type (e.g. a custom `MyModelError(RuntimeError)`), only the
`RuntimeError`-ness survives the `jit` wrapping — `except MyModelError:`
will **not** match, because the concrete type identity is lost in the
`XlaRuntimeError` wrapping. Only a bare (non-jit) `pure_callback` call
preserves the original exception object and type exactly.

### 2b. Verification

- D0 (claim 3): bare `pure_callback` raise, and raise inside `custom_vjp`'s
  `model_bwd`, both under plain `jax.grad` and `jax.vmap(jax.grad(...))` —
  confirmed the exception propagates and carries the original message,
  without yet pinning down the exact type.
- D1#8 (`claim8_raise_under_svgd_composition`): extends this specifically to
  `jax.jit(jax.grad(model))(...)` and
  `jax.vmap(jax.jit(jax.grad(model)))(...)` — the literal composition
  `SVGD._precompile_model`/`_precompile_unified` build (see §2d) — and pins
  the exact type via direct inspection:
  ```
  jit(grad(.)):        jaxlib._jax.XlaRuntimeError, isinstance RuntimeError: True
  vmap(jit(grad(.))):  jaxlib._jax.XlaRuntimeError, isinstance RuntimeError: True
  ```
  and confirms `issubclass(jaxlib._jax.XlaRuntimeError, RuntimeError) is
  True` directly.
- A non-triggering batch under the same `vmap(jit(grad(...)))` composition
  was also checked and does **not** raise, returning finite gradients — the
  raising path only fires when the host function's condition is actually
  met, i.e. the mechanism doesn't spuriously trip.

### 2c. Real codebase example

Every `model_bwd`/`_fd_theta_bar`/`_exact_*_jac_np` host callback in this
codebase that can hit a genuinely unexpected internal state raises a plain
`RuntimeError`/`ValueError` from inside its `pure_callback`-wrapped host
function and relies on exactly this propagation — e.g.
`_check_weight`/`_apply_weight_callback` (referenced in D0's docstring), and
the validation raises in `_check_negative_pmf` (`svgd.py:356–380`, via
`jax.debug.callback` — see §7, which uses the *same* propagation mechanism).

### 2d. SVGD's actual gradient-compilation composition

`SVGD._precompile_model` (`svgd.py:6352–6400`) and `_precompile_unified`
(`svgd.py:6434–6548`) both build:

```python
grad_fn = jax.grad(log_prob_fn)      # or self._log_prob
compiled_grad = jax.jit(grad_fn)     # svgd.py:6395 / :6521
```

and `svgd_step` (`svgd.py:4023–4162`) then calls it as:

```python
grad_log_p = vmap(compiled_grad_to_use)(particles_for_grad)   # svgd.py:4145
```

i.e. the actual composition at every SVGD iteration's per-particle gradient
step is **`vmap(jit(grad(f)))`** (vmap outermost, jit innermost) — not
`jit(vmap(grad(f)))` or `jit(grad(vmap(f)))`. §2b's D1#8 result is against
this exact composition, not a simplified stand-in. The `pmap(vmap(...))`
branch (`svgd.py:4137`, multi-device) nests one level further —
`pmap(vmap(compiled_grad))` — which was not independently re-verified for
exception propagation in this session (pmap requires ≥2 local devices,
impractical to set up for this check); treat its exception-propagation
behavior as *likely* the same mechanism (pmap and vmap share the batching
machinery) but **not empirically confirmed** here.

---

## 3. `jax.pure_callback`'s `vmap_method`: `'sequential'` vs `'expand_dims'`

### 3a. The rule

- **`'sequential'`**: under `vmap`, JAX calls the Python host function once
  **per batch element**, serialized, each time with that element's
  arguments **unbatched** (no extra leading axis) — as if you'd called the
  unvmapped function in a Python loop. Genuinely non-batched (constant)
  arguments are passed through completely unchanged (their original shape).
- **`'expand_dims'`**: under `vmap`, JAX calls the Python host function
  **once**, with a leading batch axis on every argument — genuinely-batched
  arguments get their real batch size on axis 0; arguments that are
  *not* actually varying across the batch (a closed-over constant) still
  get a **size-1 axis inserted** at position 0, so every argument has
  consistent extra leading dimensionality. The callback body is therefore
  **required** to loop over (or otherwise vectorize across) axis 0 itself —
  JAX does not do this for you.

D1#4 (probe script `/tmp/jax_semantics_probe4.py`, not checked into the
repo — reproducible from the description above) confirmed the mixed
batched/unbatched-argument shapes precisely: with `times` genuinely batched
and `theta` a per-call constant, `'sequential'` delivered `theta` at shape
`(1,)` (unchanged) and `times` as a scalar `()` per call (3 calls total);
`'expand_dims'` delivered `theta` at shape `(1, 1)` (size-1 axis inserted)
and `times` at its real batch shape `(3,)`, in a single call.

### 3a-bis. Does the `vmap_method` choice affect exception propagation?

No — §2's exception-propagation mechanism is orthogonal to `vmap_method`.
Both D0 (claim 3, `'sequential'`) and D1#8 (`'sequential'`, inside a
`custom_vjp` `bwd`) confirm a raise propagates correctly; nothing about
`'expand_dims'`'s single-batched-call shape changes *whether* an exception
raised inside the (one, batched) callback invocation reaches the caller —
it's the same `pure_callback`-boundary mechanism either way. What
`vmap_method` changes is **only** the shape/count of calls into the host
function (§3a), and therefore *when within the batch* a given raise
condition gets checked (all-at-once for `'expand_dims'`, one element at a
time — so potentially after fewer completed elements — for
`'sequential'`), not whether the raise itself survives the JAX boundary.

### 3b. Correctness consequence for a stateful callback (this codebase's actual use case)

Every FFI-adjacent host callback in this codebase that needs to mutate then
read a private, stateful C++ object (`graph.update_weights(theta)` then
read a result) needs one of:
- `'sequential'` — safe unconditionally: each call is a clean
  update-then-read on an unbatched row, with **no risk of cross-particle
  interleaving** (D1#9 confirmed the call log alternates
  `update(s), read(s)` pairs in lockstep, never `update, update, read,
  read`), or
- `'expand_dims'` **with the callback explicitly looping over the received
  batch axis** — also safe, but only if the callback author remembers the
  loop.

D1#9 (`claim9_vmap_method_stateful_callback`) demonstrates the concrete
**failure mode** of getting `'expand_dims'` wrong: a callback written as if
it always receives a single unbatched row (the same code that's correct
under `'sequential'`) receives the *entire batch* in one call under
`'expand_dims'` and, having no loop, updates the state only from the
batch-reduced input and returns the **same stale value broadcast to every
particle** — `[24.0, 24.0, 24.0]` instead of `[4.0, 8.0, 12.0]`. Critically,
**this does not raise** — it is a silent wrong-answer failure, not a crash,
which makes it far more dangerous than a shape mismatch.

### 3c. Real codebase examples, catalogued

| Site | vmap_method | Rationale (from comment, or inferred from the callback body) |
|---|---|---|
| `:3662–3673` `model_pure` (weight-callback mode) | `'sequential'` | **DEAD CODE** — inside `pmf_from_graph_parameterized`, which raises `NotImplementedError` at `:3896` before this code is ever reached (CLAUDE.md "Disabled paths"). Each theta needs a distinct `weight_callback` application + full graph rebuild — could not be batched even if live. |
| `:4174–4183` `_compute_pmf_pure` (`pmf_from_cpp`) | `'expand_dims'` | See §3d — shares `_compute_pmf_from_ctypes` (`:423–496`) with `moments_from_graph`'s helper; **not independently confirmed batch-safe** in this session (see §3e). |
| `:6781–6786` `_compute_pure` (`moments_from_graph`) | `'expand_dims'` | **CONFIRMED BROKEN under a genuine batched theta — see §3d.** |
| `:7134–7138` `_compute_pure` (`pmf_and_moments_from_graph`, callback weight-mode) | `'sequential'` | Each theta needs a distinct `weight_callback` application + full `GraphBuilder` rebuild (mirrors `:3662`'s live rationale). |
| `:7186–7191` `_cdf_zero_pure_cb` (callback-mode cdf_zero) | `'sequential'` | Same weight_callback constraint as above. |
| `:7391–7396` `_compute_pure` (`pmf_and_moments_from_graph`, pybind/GraphBuilder mode) | `'expand_dims'` | `callback_fn` (`:7364–7389`) **explicitly branches on `theta_np.ndim == 2`** and loops via `_compute_pmf_and_moments_cached` (`:7291–7320`, its own `if theta_np.ndim == 2: for theta_single in theta_np: ...` loop) — correctly handles the batch. |
| `:7466–7471` `_cdf_zero_pure` (pybind-mode cdf_zero) | `'expand_dims'` | Same `_compute_cdf_zero_cached` ndim==2 loop pattern (`:7415–7444`). |
| `:7568–7571` `_exact_moments_jac_np` (exact moment-gradient Jacobian) | `'expand_dims'` | Explicitly checks `if th.ndim == 1: return _one(th)` else loops `for _b in range(th.shape[0]): out[_b] = _one(th[_b])` (`:7035–7040`) — correctly handles the batch. |
| `:8026–8030`, `:8057–8061` `_compute_pure` (`pmf_from_graph_joint_index`, callback mode, baked/non-baked) | `'sequential'` | Same weight_callback constraint. |
| `:8248–8254` `_exact_sojourn_jac_np` (exact sojourn-gradient Jacobian, in `model_bwd`) | `'sequential'` | Mutates the private clone graph (`_jix_exact_graph.update_weights(...)`) then reads — the stateful-object pattern from §3b; `'sequential'` sidesteps the need for an internal batch loop entirely. |

### 3d. Real codebase example used incorrectly — confirmed by direct execution

`Graph.moments_from_graph`'s pure_callback (`:6781–6786`, `vmap_method=
'expand_dims'`) wraps `_compute_moments_pure` (`:6766–6779`), which calls
into a JIT-compiled ctypes library assuming **`theta_flat` is always 1-D**
(`len(theta_np)` as the parameter count, no `ndim` check at all) — unlike
its three siblings in the table above (`_compute_pmf_and_moments_cached`,
`_exact_moments_jac_np`, `_compute_cdf_zero_cached`), all of which
explicitly detect and loop over a 2-D (batched) input.

D1#13 (`claim13_moments_from_graph_vmap_break`, run with
`PHASIC_SOURCE_DIR=/Users/kmt/phasic` since this path JIT-compiles C++ from
source) confirms this **actually breaks** — not just theoretically — when
`moments_from_graph`'s output is vmapped over a genuinely-batched theta:

```
jax.vmap(moments_fn)(theta_batch)
  -> RuntimeError: Incorrect output shape for return value #0:
     Expected: (3, 1), Actual: (1,)
```

JAX's own callback-output-shape check catches the mismatch (it expected the
callback to return a `(batch, nr_moments)` array under `'expand_dims'` but
got the unbatched `(nr_moments,)` shape back), so this fails loudly rather
than silently — a better outcome than §3b's silent-wrong-answer scenario,
but still a genuine gap: **`Graph.moments_from_graph` cannot currently be
`jax.vmap`'d over theta at all.** This is new information beyond what
CLAUDE.md's "Disabled paths / follow-ups" already documents for this
function (it flags `moments_from_graph` as "untouched by the exact-grad
work" / FD-only, but does not mention that the forward pass itself — no
gradient involved — breaks under `vmap`).

### 3e. Related, structurally similar, NOT independently confirmed

`Graph.pmf_from_cpp`'s `_compute_pmf_pure` (`:4174–4183`) is built on
`_compute_pmf_from_ctypes` (`:423–496`), which has the **same** unconditional
`len(theta_np)` / no-`ndim`-check pattern as `moments_from_graph`'s helper.
`pmf_from_cpp` is a **public, documented, live** classmethod (docstring
example at `:4063`), so if a caller `jax.vmap`'d it directly over a genuine
theta batch, the same class of failure seems structurally likely. This was
**not executed in this session** (would require authoring a compilable
`build_model()` C++ file) — flagging it as a probable-but-unverified
sibling gap rather than a confirmed one, per this repo's
"ground code claims" discipline. Its one exercised call site
(`non_param_wrapper`, `:3836–3840`, itself inside the disabled
`pmf_from_graph_parameterized`) always passes a **constant** `dummy_theta`
that is never the vmapped argument, so that particular call path is not
exposed to the bug even if it exists.

---

## 4. `jax.experimental.custom_batching.custom_vmap`

### 4a. The rule

`custom_vmap` lets a function register an **explicit** batching rule
(`@f.def_vmap`, signature `(axis_size, in_batched, *batched_args) ->
(outputs, out_batched)`) that JAX calls **instead of** its default
per-argument auto-batching whenever the function is used under `vmap` — in
*any* composition, including calls made from inside a `jax.custom_vjp`'s
`fwd` or `bwd`. This matters when the default auto-batching would be wrong
or unsupported for the wrapped operation — e.g. an FFI/host boundary that
hard-rejects a 3-D buffer, where naive auto-batching would add a naive
extra leading axis on top of an already-2-D call.

### 4b. Verification

D1#10 (`claim10_custom_vmap_fires_inside_custom_vjp_bwd`): a `core` function
wrapped in `custom_vmap` (asserting its callee only accepts 2-D input),
itself wrapped in a `jax.custom_vjp` with a hand-rolled central-difference
`bwd` that calls `core` repeatedly. Under `vmap(jax.grad(autodiff))`, the
`custom_vmap` rule fired **5 times** (once for the plain forward path, four
more for the FD taps inside `bwd`) — confirming the rule intercepts calls
made from inside both `fwd` and `bwd`, not just a bare top-level `vmap(core)`
call, fusing what would otherwise be many small per-particle calls into
fused batched calls.

### 4c. Real codebase examples

- `_daisy_chain_svgd_model`'s `_forward`/`_forward_vmap_rule` (no-exposure
  branch, `:4642–4666`) and `_per_obs_core`/`_per_obs_core_vmap_rule`
  (exposure branch, `:4857–4893`): each fuses what would otherwise be `P`
  separate per-particle FFI calls of shape `(theta_dim,)` into a single
  `(P, theta_dim)` (or `(P*n_unique, theta_dim)`) call, exploiting exactly
  the mechanism D1#10 confirms — the custom_vjp `bwd`'s FD perturbation
  loop (`_autodiff_bwd`/`_per_obs_bwd`) calls `_forward`/`_per_obs_core`
  repeatedly, and every one of those calls goes through the fused rule.
- `pmf_from_graph_joint_index`'s `_obs_forward`/`_all_forward` (baked
  `observed_indices` branch, `:8077–8111`): same fusion pattern for the
  sojourn-times FFI call.

### 4d. Real codebase caveat — when NOT to add a `custom_vmap` rule

`daisy_chain_joint_probs` (`:10084–10363`) deliberately does **not** wrap
its `_forward` in a `custom_vmap` rule, and the comment at `:10298–10311`
explains why: that function's `_forward` closes over
`structure_json_str`/`initial_ipv_arr`, which are **rebuilt on every call**
to `daisy_chain_joint_probs` — i.e. they are concrete only relative to
*that* call, not truly fixed. Under `vmap(jit(grad(...)))`, a `custom_vmap`
rule's closure over these values was found to leak an **enclosing-trace
tracer** into the rule's inner jaxpr as a `DynamicJaxprTracer` constant (in
some execution contexts, notably `ipykernel`). `Graph._daisy_chain_svgd_
model` (§4c) sidesteps this by building its `_forward` + rule once, at
**model-construction time**, outside any trace, so its closures are
genuine concrete arrays. **Rule of thumb: a `custom_vmap` rule's closed-over
values must be built outside any active trace** (at model-construction
time, not inside a function JAX will later trace/jit/grad) or the rule
itself can silently become the source of a tracer leak.

---

## 5. Static vs traced control flow: when a bare Python `if` is safe

### 5a. The rule, precisely

A bare Python `if <value>:` inside a function JAX will trace (via `jit`,
`grad`, or `vmap`) is safe **if and only if** `<value>` is a concrete Python
object at the moment the `if` executes — a plain `bool`/`int`/`float`, or a
JAX/NumPy array with no tracer anywhere in its construction history. In
practice this means: `<value>` was computed **before** the enclosing
closure is ever handed to `jit`/`grad`/`vmap` (i.e. it is fixed once, at
Python/model-construction time, and merely *closed over* by the traced
function), and is never itself derived from a traced input or from a
`pure_callback`'s **output** (which, from the tracer's perspective, doesn't
have a value yet at trace time — it's a promise of a future runtime call,
not a Python value).

If this rule is violated, JAX raises `jax.errors.TracerBoolConversionError`
("Attempted boolean conversion of traced array...") — but **only** once the
function is actually traced. D1#12
(`claim12_static_vs_traced_if`) confirmed the exact same function, called
**eagerly** (no `jit`/`grad`/`vmap` at all), runs with **no error** — eager
arrays are concrete, so `if` on them works exactly like NumPy. **This means
the bug is invisible in quick interactive/eager testing and only surfaces
once the function is wrapped for real use** (the whole point of building
these JAX-compatible model wrappers in the first place).

### 5b. Corollary: static Python loops over concrete shapes are fine, even with data-dependent-looking membership tests

`for i in range(n_params): if i in fixed_indices_set: continue` — used
throughout this codebase's FD backward loops (e.g. `:7580–7586`,
`:8203–8207`, `:4679–4682`, `:4919–4922`) — is safe even though it sits
directly inside a traced `model_bwd`, because `n_params = theta.shape[0]`
is a **static shape** (known at trace time even for a tracer — shapes are
never traced, only values are) and `fixed_indices_set` is a plain Python
`set` built from `fixed_mask`/`fixed_indices` **before** tracing. `i` is
therefore always a concrete Python `int`; the `for` loop **unrolls** at
trace time (n_params separate traced code paths get built, not a runtime
loop), and `i in fixed_indices_set` is pure Python, never touching a
tracer. This is a completely different situation from branching on a value
*inside* an array (§5a) — the rule is about what the branched-on *value*
is, not about whether the branch textually sits inside a traced function.

### 5c. Real codebase example — correct, exactly the pattern this document was asked to explain

`_jix_exact_enabled` (`pmf_from_graph_joint_index`, `:7868–7916`) is
computed from `graph.get_was_dph()`, `graph._weight_mode`,
`graph.param_length()`, and the `exact_grad` kwarg — **all Python-level,
concrete values available at model-construction time**, before the
`model`/`model_fwd`/`model_bwd` closures are ever handed to
`jax.grad`/`jax.jit`/`jax.vmap`:

```python
if not bool(exact_grad):
    _jix_exact_enabled = False
elif not _jix_linear_scope_ok:
    _jix_exact_enabled = False
elif _jix_was_dph:
    _jix_exact_enabled = False
elif _baked:
    _jix_exact_enabled = False
elif int(graph.param_length()) != param_length_actual:
    _jix_exact_enabled = False
else:
    _jix_exact_enabled = True
```

This bare-`if` chain decides which of two **Python code paths get built**
into the closure returned by `pmf_from_graph_joint_index` (`:7918–7961`
defines `_exact_sojourn_jac_np` only `if _jix_exact_enabled`), then later
`model_bwd` branches on it again with a bare `if _exact_sojourn_jac_np is
None:` (`:8217`) to choose which branch of Python code to even trace in the
first place — not a runtime `lax.cond` choice at all. This is exactly why
it's safe: `_jix_exact_enabled` never flows through `jax.grad`/`vmap`
tracing as a value; it decides, once, in plain Python, which computation
graph gets built.

### 5d. Real codebase example — the traced counterpart that correctly does NOT use a bare `if`

`exact_ok` (same function, `:8255,8268`) is a fundamentally different kind
of value — it is `jnp.all(jnp.isfinite(J_obs)) & jnp.all(jnp.isfinite(J_all))`
where `J_obs`/`J_all` are the **output of a `pure_callback`** (`:8248–8254`).
Under `jit`/`grad`/`vmap`, `exact_ok` is a genuine tracer whose value isn't
known until the callback actually runs at runtime — exactly the case §5a
forbids branching on with a bare `if`. The code correctly uses
`jax.lax.cond(exact_ok, lambda: exact_tbm, _fd_theta_bar)` (`:8268`)
instead — see §1e.

---

## 6. `jax.vmap` itself: what gets batched, what's held constant, in this codebase

### 6a. The dominant pattern: vmap over the particle axis

Every SVGD gradient step vmaps over the **leading (particle) axis of
theta**, and nothing else — `svgd_step`'s `vmap(compiled_grad_to_use)
(particles_for_grad)` (`svgd.py:4145`) and `vmap(grad(log_prob_for_grad))
(particles_for_grad)` (`svgd.py:4147`). Observation/time data
(`observed_data`), the pre-built `structure_json`/FFI-serialized graph, and
the source `Graph`/private clone objects are all **closed over as
construction-time constants** — never arguments to the vmapped function,
never batched. This is precisely what makes the `custom_vmap`
fusion tricks in §4 possible: only theta genuinely varies across the vmap
axis, so a rule that fuses `(P, theta_dim)` into one FFI call is sound.

Other direct `vmap(log_prob_fn)(self.particles)` call sites follow the same
pattern (posterior-diagnostics helpers, `svgd.py:8109,8339,8359`).

### 6b. The deliberate exception: `jax.lax.map`, not `vmap`, for per-observation dispatch

`_wrap_model_with_exposure` (`svgd.py:257–338`) needs to evaluate the base
model once per **observation**, each with its own scaled theta, i.e. a
*paired* `(theta_i, time_i)` iteration — and deliberately uses
`jax.lax.map`, not `jax.vmap`:

```python
pmf_per_obs, moments_per_obs = jax.lax.map(per_obs, (theta_batch, times))
```

The comment (`svgd.py:320–325`) states why: "`lax.map` runs the body
sequentially under JIT — this matches the underlying model's
`pure_callback`/FFI batching contract, which does not support paired
`(theta_i, time_i)` batching." `lax.map`'s single-Python-trace-call
behavior (confirmed: the body function is traced exactly once regardless
of the number of elements, since `lax.map` compiles to `lax.scan`) means it
cannot be used the same way as the counting-based probes in §1/§3 to prove
sequential *execution* — that claim rests on JAX's own documented `lax.map`
→ `lax.scan` lowering (sequential by default, one carry step at a time)
rather than an independent empirical timing check in this session.

### 6c. Multi-device composition: `pmap(vmap(...))`

`svgd_step`'s `parallel_mode='pmap'` branch (`svgd.py:4137,4139`) composes
`pmap(vmap(compiled_grad_to_use), axis_name="batch")` — outer `pmap` shards
the particle batch across devices, inner `vmap` batches within each
device's shard. This was not independently re-verified for the §1/§2 rules
in this session (would need ≥2 local devices) — treat its `lax.cond`/
exception-propagation behavior as *likely* identical to the `vmap`-only
case (pmap's batching is built on the same machinery) but not empirically
re-confirmed here.

---

## 7. Related pattern found along the way: `jax.debug.callback`

Not explicitly asked for, but used in this codebase for a genuinely
different purpose than `pure_callback`, and worth distinguishing precisely
since both live in `svgd.py`'s validation path.

### 7a. The rule

`jax.debug.callback` is **fire-and-forget**: its host function's return
value (if any) is discarded and never flows back into the traced
computation — unlike `pure_callback`, whose whole purpose is to feed a
value (a `ShapeDtypeStruct`-typed result) back into the trace. This makes
`jax.debug.callback` the right tool for pure side effects (logging,
assertions) where no computed value is needed downstream.

**However — and this was not obvious in advance — a raise inside
`jax.debug.callback`'s host function propagates to the caller exactly the
same way a `pure_callback` raise does.** D1#11
(`claim11_debug_callback_exceptions_propagate`) confirmed this empirically:
under `jax.jit`, the exception surfaces as the same
`jaxlib._jax.XlaRuntimeError` (§2a) with the original message embedded.
"Fire-and-forget" describes the *return value*, not error handling — a
natural (and here, wrong) assumption would be that a callback with no
return value might also have its errors silently discarded; it does not.

### 7b. Real codebase examples

- **Raises**: `_check_negative_pmf`/`_check_sparse`/`_check_dense`
  (`svgd.py:356–380`, `6034–6044`, `6099–6108`) validate that the model
  never returned a NaN or negative "probability" and raise `ValueError` if
  it did — deliberately surfacing the *real* cause (an invalid rate) at the
  point it happens, rather than letting a NaN silently poison the gradient
  and only surface an iteration later as an opaque failure.
- **Does not raise (pure logging)**: `_warn_grad_norm_clipped`
  (`svgd.py:4007–4020,4162`) logs a `logger.warning(...)` when the
  gradient-norm clip (`_GRAD_NORM_CLIP_MULT`) actually clipped a particle —
  a deliberately non-fatal notification, matching this codebase's
  no-silent-fallback principle without aborting the run.

Both patterns coexist correctly: `jax.debug.callback` is used exactly where
no value needs to flow back into the trace, and the raising variant relies
on (and gets) the same robust exception propagation as `pure_callback`.

---

## Checklist to run past any new JAX-wiring design in this codebase

1. **Does a `lax.cond` (or a value that will end up guarding one) ever sit
   inside a function this codebase will `vmap` over particles?** If so, its
   predicate WILL be batched, both branches WILL execute in full on every
   call, and the "skip" you may be relying on for cost does not exist under
   vmap. Ask: is there a genuinely cheaper alternative that's only skippable
   *outside* vmap (§1e), or does the "cheap" branch's prerequisite work run
   unconditionally anyway (§1f, favor `jnp.where` then)?
2. **Is the predicate/condition value a concrete Python object computed
   before the closure is handed to `jit`/`grad`/`vmap`, or is it derived
   from a traced input / a `pure_callback`'s output?** The former: a bare
   `if` is safe and should be preferred (it lets vmap skip the unused
   branch entirely, §D0 claim 2). The latter: use `lax.cond`/`jnp.where`,
   and don't trust an eager smoke-test to catch a violation (§5a) — test
   under actual `jit`/`grad`/`vmap`.
3. **Does a `pure_callback`'s host function ever raise?** If it needs to
   propagate to the caller as a real exception, that already works in every
   composition tested (§2a) — but if you plan to `except SomeSpecificType:`
   at the call site, check whether the composition includes `jit`
   (`XlaRuntimeError` wrapping loses everything but `RuntimeError`-ness and
   the message text, §2a).
4. **Does the callback mutate a private, stateful object (`graph.
   update_weights(...)` then read)?** Prefer `vmap_method='sequential'`
   (each call is a clean update-then-read, no interleaving risk, §3b) —
   or, if using `'expand_dims'` for its single-call efficiency, verify the
   callback body actually loops/vectorizes over the received leading axis
   for EVERY argument, including ones that "feel" constant (they still get
   a size-1 axis inserted, §3a). A missing loop fails **silently** with
   wrong values, not a crash (§3b, §3d) — write a value-correctness test
   under `jax.vmap` specifically, not just a shape/`isfinite` check.
5. **Is a `custom_vmap` rule's closure built from values that are only
   concrete relative to a single call (rebuilt every invocation), or truly
   fixed at model-construction time?** The former risks a tracer leak
   under `vmap(jit(grad(...)))` in some execution contexts (§4d) — build
   `custom_vmap`-wrapped functions and their rules once, outside any trace.
6. **What varies across the `vmap` axis, and what's actually a
   per-call-constant closure?** Get this explicit before reaching for
   `custom_vmap` fusion (§4a) or worrying about `vmap_method` (§3a) —
   both depend entirely on which arguments are genuinely batched.
7. **If sequential (not batched) per-element dispatch is actually required**
   (e.g. paired `(theta_i, time_i)` where the batching contract can't
   support it), reach for `lax.map`, not `vmap` (§6b) — and don't expect a
   Python-level call-counter to prove its sequential-*execution* claim
   either (it traces the body once, same caveat as `lax.cond`, §6b).
