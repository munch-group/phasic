# Audit — Phase 1: Forward-value parity

**Verdict: the claim under test is CONFIRMED.**

> *Claim:* the refactor changed only gradient (backward-pass) code and added raise-only
> guards, therefore every FORWARD value must be bit-identical to `git HEAD` outside the
> three paths whitelisted in §8 of the hand-off.

Across **212 cells** of the public compute surface, evaluated independently against the
refactored working tree and a separately-built `git HEAD` worktree:

| | |
|---|---|
| cells that ran successfully on **both** sides | **177** |
| of those, **bit-identical** (float64 hex, not a tolerance) | **177 / 177** |
| forward-value differences outside the whitelist | **0** |
| cells that worked on HEAD and now raise (regressions) | **0** |
| cells that raised on HEAD and now work | **26** — all inside the §8 whitelist |
| cells that raise on both sides | **2** — `pmf_from_graph_parameterized` (bug 5, still broken) |

**No regressions were found.**

However, the audit surfaced **six defects that are not forward-parity regressions** but
are real, and one of them (**F1**) says a stated premise of the refactor is *false*, and
another (**F2**) says the refactor makes a **silently-wrong-answer path reachable for the
first time**. Those are in §4 and must be read before sign-off.

---

## 1. What was compared, and why the comparison is trustworthy

The single biggest risk in this phase is a **fake** result — two sides that differ for an
environmental reason, or that agree because they were secretly the same build. Each of
those was closed explicitly:

| hazard | control |
|---|---|
| HEAD build contaminated by the working tree | `git worktree add --detach` to a temp dir. **No `git stash`, no `git checkout -- .`** — the uncommitted notebooks in the tree were never touched (verified: `git status` unchanged). |
| the two sides sharing one compiled extension | Each side loads **its own** `phasic_pybind*.so`. Verified by sha1: `811124233310` (refactor) vs `1cad5a73cb42` (HEAD). The C++ diff is inside that extension, so they *must* differ. |
| the two sides sharing one Python package | refactor = the normal `site-packages` install; HEAD = a separate `pip install --target` dir on `PYTHONPATH`. Verified: `_fd_probe_points` present on one side, absent on the other. |
| **XLA-FFI silently unavailable on one side** → it would fall back to slow Python callbacks and every value would differ for the wrong reason | `XLA_FFI_INCLUDE_DIR` set for **both** builds; `_register_ffi_targets()` returns `True` on **both**. This was checked, not assumed. |
| **a shared on-disk cache letting one side read the other's numbers** | separate `PHASIC_CACHE_DIR` per side. |
| JIT `.so` collision between sides | none possible: `_secure_artifact_dir()` (`__init__.py:580`) is a fresh per-process `mkdtemp`. |
| thread-count-dependent reduction order | `OMP_NUM_THREADS=1`, identical on both sides. |
| stale install | both sides rebuilt from source immediately before the sweep. |

Both sides were built inside the **same pixi environment activation**, so compiler,
MPFR/GMP, OpenMP and JAX are identical; the *only* difference is the source.

Comparison is on the **float64 hex** of every output element — bit-identity, not a
tolerance. Exceptions are compared by type, with absolute paths normalised out.

### Reproduce

```bash
S=<scratch>                                   # harnesses live here
$S/run_side.sh new     # working tree -> forward_new.jsonl
$S/run_side.sh head    # git HEAD     -> forward_head.jsonl
python3 $S/diff.py     # bit-exact diff + regression classification
python3 $S/deadflags.py new   # within-side inert-flag analysis
```

---

## 2. Results by entry point

| entry point | cells | HEAD ok | refactor ok | bit-identical | value diffs | new raises |
|---|---|---|---|---|---|---|
| `daisy_chain_joint_probs` | 16 | 16 | 16 | 16 | 0 | 0 |
| `joint_index` | 16 | 16 | 16 | 16 | 0 | 0 |
| `joint_prob_graph` (structure) | 2 | 2 | 2 | 2 | 0 | 0 |
| `joint_sojourn_graph` (structure) | 2 | 2 | 2 | 2 | 0 | 0 |
| `joint_stop_prob_graph` (structure) | 2 | 2 | 2 | 2 | 0 | 0 |
| `moments_from_graph` | 24 | **0** | 24 | — | 0 | 0 |
| `pmf_and_moments_from_graph` | 64 | 64 | 64 | 64 | 0 | 0 |
| `pmf_and_moments_..._multivariate` | 16 | 16 | 16 | 16 | 0 | 0 |
| `pmf_from_cpp` | 2 | **0** | 2 | — | 0 | 0 |
| `pmf_from_graph` | 32 | 32 | 32 | 32 | 0 | 0 |
| `pmf_from_graph_parameterized` | 2 | 0 | **0** | — | 0 | 0 |
| `reward_visit_probability` | 12 | 12 | 12 | 12 | 0 | 0 |
| `SVGD.log_likelihood` | 13 | 6 | 6 | 6 | 0 | 0 |
| `Graph.pdf/cdf/moments/…` (C core) | 7 | 7 | 7 | 7 | 0 | 0 |
| `svgd log-lik expression` | 2 | 2 | 2 | 2 | 0 | 0 |
| **TOTAL** | **212** | **177** | **203** | **177** | **0** | **0** |

### The 26 newly-working cells — all whitelisted

All 26 raised `RuntimeError: Compilation failed` on HEAD (bug 3, `_compile_wrapper_library`)
and now produce values:

* `moments_from_graph` × 24 — and the values are **correct**: `E[T^k]` on the
  hypoexponential(8,5) test chain returns `[0.325, 0.16125, 0.10846875, 0.09263437]`,
  matching the independent `Graph.moments()` C++ reference to the last bit, and
  `E[T] = 1/8 + 1/5 = 0.325` in closed form. (Bug 2's `(n+1)!` check also reproduces:
  `[2, 6, 24, 120]` on Erlang(2,1).)
* `pmf_from_cpp` × 2 — discrete now returns a genuine NegBinomial-shaped PMF
  `[0.26, 0.247, 0.1192, 0.0104]` rather than the collapsed deterministic walk (bug 4).

### The 2 cells that raise on both sides

`pmf_from_graph_parameterized` (bug 5) — still broken, exactly as the hand-off admits
(§6 gap 4). The *exception type* changed (HEAD: `RuntimeError: Compilation failed` →
refactor: `XlaRuntimeError`/`CpuCallback`) **because the compile now succeeds and it fails
one step later**, at the `jnp.float32` / `NoneType.ShapeDtypeStruct` defect. Not a
regression: broken before, broken after.

### SVGD

`Graph.svgd()`'s particle **trajectory** is out of scope for bit-parity *by construction* —
it consumes the gradients, which the refactor intentionally changed. What must be
identical is the forward likelihood it evaluates, which is precisely the code `svgd.py`
touched (`_log_lik_from_pmf`, `:5850`). `SVGD.log_likelihood(theta)` at a **fixed** theta
is that forward value. All 6 runnable cells are **bit-identical**, and they cover both
branches the guards were wired into:

* dense branch (`:5951`) — linear / log / formula, and `rewards=full`
* **sparse** branch (`:5887`) — `sparse=True`
* **zero-inflated** branch — `rewards=partial` returns a distinct `0.17545338158076795`
  (vs `2.667970971773137`), proving that branch was genuinely exercised rather than
  skipped.

---

## 3. The structural claim, verified by reading the whole diff

§8 of the hand-off asserts every changed line is backward-pass, build machinery, or
raise-only. I read all four changed source files end-to-end. **The assertion holds**, and
the C++ file — the one piece §8 does not itemise — holds too:

| file | change | forward-safe? |
|---|---|---|
| `svgd.py` | `_check_negative_pmf` + two `jax.debug.callback` guards; `1e-10` → `_PMF_LOG_OFFSET` | ✅ guards can only raise; `_PMF_LOG_OFFSET == 1e-10` is the **same number**, so `log(pmf + …)` is unchanged |
| `ffi_wrappers.py` | `_rvp_bwd` only, plus `_weight_mode_of` helpers | ✅ backward-only |
| `__init__.py` | `_fd_probe_points` (called **only** inside `*_bwd`), `_compile_wrapper_library`, two C++ wrapper strings | ✅ the wrapper strings are the whitelisted JIT paths |
| `graph_builder_ffi.cpp` | +113/−0: `kMinEdgeWeight`, `most_negative_edge_weight`, checked in both daisy handlers | ✅ a **read-only** min-reduction over edge weights; writes NaN and returns `ffi::Error` **only** when a weight is `< -1e-12`. No valid-input value is touched. |

The `fd_nonneg = getattr(graph, '_weight_mode', …) == 'linear'` lines sit in
forward-constructing scope but are **read only by the backward rule** — confirmed by the
sweep (16/16 `use_cache` pairs and every `fixed_mask` pair are forward-identical).

---

## 4. FINDINGS — defects that are *not* parity regressions

These are real. None of them breaks forward parity, so none blocks the "no regression"
verdict — but **F1 and F2 should block sign-off** until addressed.

### F1 — HIGH — the `nonneg=False` rule for `weight_mode='log'` rests on a FALSE premise

The refactor's own comment (`__init__.py:732`) and §3 of the hand-off both say:

> `'log'` (theta is a log-scale, legitimately negative) → `nonneg=False`

**This is factually wrong.** In `log` mode the weight is a **product**, computed in
log-space:

* `__init__.py:1765` — "`'log'`: weight = Π(c_k θ_k) (computed in log-space for stability)"
* `phasic_pybind.cpp:1494` — `edge.weight = exp(sum(log(cᵢ*θᵢ))) = (c₁*θ₁) * … * (cₙ*θₙ)`,
  and its own docstring: *"All (cᵢ*θᵢ) products must be positive when log=True … Raises
  RuntimeError if any (coefficient * parameter) product is non-positive."*

Verified by execution:

```
update_weights([1.0,  2.0], log=True) -> w = 12.0
update_weights([1.0, -2.0], log=True) -> RuntimeError: ... products to be positive. Got -6.0
update_weights([1.0,  0.0], log=True) -> RuntimeError: ... products to be positive. Got 0.0
```

So under `log`, θ must be **strictly positive** — *stricter* than `linear`, which tolerates
θ = 0. `log` is the one mode where the positivity floor is most needed, and it is exactly
the mode the refactor turned the floor **off** for.

*Practical impact is narrow but real — the exact boundary, measured:*

```
weight_mode='log', jax.grad as theta[1] shrinks:
  theta[1]=1e-08  -> grad ok                       (HEAD raised here: |step| 1e-7 > theta)
  theta[1]=1e-12  -> grad ok
  theta[1]=1e-15  -> ValueError: log weight mode requires ... products to be positive
  theta[1]=0      -> ValueError  (forward is ALSO invalid here: c*0 = 0 is non-positive)
```

So the relative step **did** fix `log` — it moved the failure from θ < 1e-7 (HEAD) to
θ ≤ 1e-15, an eight-order-of-magnitude improvement — it just documented a false reason for
doing so and left a residual hole. At θ = 1e-15 the **forward is valid but the gradient
raises**: that is purely an FD-probe defect. Below that, `log` rejects the parameter anyway.

*Mitigating:* `log` requires **every** `(cᵢ·θᵢ) > 0`, so it is unusable on any graph with a
zero coefficient — i.e. the whole coalescent/joint-prob family. `log` only applies to
dense-coefficient graphs, which narrows who can hit this at all.

**Fix:** `nonneg` should be `True` for `'log'` as well as `'linear'`. More importantly, the
comment and the hand-off must be corrected — as written they will mislead the next
maintainer into thinking negative θ is legitimate under `log`, when the library rejects it.

### F2 — HIGH — `moments_from_graph` silently ignores `log`, `callback` AND `formula`

Re-tested with a genuinely **non-linear** weight rule (`c0*t0*t1`), so that "linear" and
"callback/formula" must give different weights. (My first test used `dot(θ,c)` and
`c0*t0 + c1*t1`, both of which are *identical to linear* — that test could not have detected
this and its agreement proved nothing. Corrected below.)

Test chain, θ=[1,2]: linear rates (8, 5) → E[T]=0.325. Non-linear rule → rates (4, 2) →
E[T]=0.75.

```
[control] pmf_from_graph(linear  ) -> 0.85082705     # honours the mode...
[control] pmf_from_graph(callback) -> 0.93161668     # ...proving the rule really differs
[control] pmf_from_graph(formula ) -> 0.93161668

moments_from_graph(linear  ) -> E[T] = 0.325   (correct)
moments_from_graph(callback) -> E[T] = 0.325   <-- WRONG; truth is 0.75
moments_from_graph(formula ) -> E[T] = 0.325   <-- WRONG; truth is 0.75
moments_from_graph(log     ) -> E[T] = 0.325   <-- WRONG; truth is 0.333
```

**Root cause:** `moments_from_graph` JIT-generates a C++ `build_model` whose weight
computation is a hardcoded linear dot product; the graph's `weight_mode` never reaches it.
(Contrast `pmf_from_graph`, which routes through `GraphBuilder`, and *that* honours the mode
— `graph_builder.cpp:29-36`, `:346`.)

**Severity is driven by `formula`/`callback`, not `log`.** `weight_formula` is a documented,
mainstream feature with its own test suite, and unlike `log` it works fine on the
sparse-coefficient graphs phasic actually uses. Any formula/callback graph fed to
`moments_from_graph` gets a **silently wrong number** — no error.

**Why this matters for *this* refactor:** on HEAD `moments_from_graph` could not compile at
all, so the wrong answer was unreachable. By fixing bug 3, **the refactor makes this
wrong-answer path reachable for the first time.** Not a parity regression (there is no HEAD
value to regress from), but a new user-visible hazard introduced by shipping this change.

### F3 — MEDIUM (downgraded) — `pmf_from_graph_joint_index` ignores `log` only

**Correction to an earlier over-claim in this report:** `joint_index` *does* honour
`callback` and `formula` (it has an explicit callback branch, `__init__.py:7325`). Re-tested
with the non-linear rule above:

```
joint_index(linear  ) -> [0.625, 1.0]
joint_index(callback) -> [0.5,   1.0]    # honoured
joint_index(formula ) -> [0.5,   1.0]    # honoured
joint_index(log     ) -> [0.625, 1.0]    <-- IGNORED: identical to linear
```

**Root cause:** `ComputeSojournTimesFfiImpl` calls `ptd_graph_update_weights(...,
/*use_log=*/false)` directly (`graph_builder_ffi.cpp:887`, `:941`), bypassing `GraphBuilder`
and hardcoding linear.

**The same hardcode is in both daisy handlers** — `:1528`
(`DaisyChainJointProbsFfiImpl`), `:1782` and `:1827` (`DaisyChainSojournFfiImpl`, the
**default** `final_read='sojourn'` path). *Read from source; NOT empirically reproduced* —
the coalescent JSP graph carries zero coefficients (`[pair_count, 0]`), and `log` requires
every `(cᵢ·θᵢ) > 0`, so it raises at build before the handler is reached.

**Practical severity is limited** by that same fact: `log` is unusable on any graph with a
zero coefficient, which is the entire coalescent/joint-prob family. It only bites on
dense-coefficient graphs.

### F4 — MEDIUM — `moments_from_graph(use_ffi=…)` is a dead flag (and it widens the §8 whitelist)

`:6441` (`if not use_ffi:`) is the flag's **only** use, and it merely gates an eager
`_ensure_jax_active()`; `:6451` imports jax unconditionally. It never selects a backend —
both settings take the same `_compile_wrapper_library` JIT path.

Two independent proofs: HEAD raised the **identical** `Compilation failed` for
`use_ffi=True` *and* `False`; and the refactor returns bit-identical values for both across
12/12 pairs.

**Consequence for the audit:** the hand-off implies `use_ffi=True` "routes elsewhere". It
does not. The §8 whitelist must cover `moments_from_graph` **in full**, not just
`use_ffi=False`. (Corrected in this report's diff.)

### F5 — MEDIUM — `pmf_from_graph(use_cache=…)` is a dead flag with a docstring that promises otherwise

Never read in the body — it appears only in the signature (`:3478`) and the docstring,
which actively advertises behaviour that does not exist (`:3501`, `:3551-3552`):

> `>>> model = Graph.pmf_from_graph(g, use_cache=True)  # First call: computes and caches`
> `>>> model2 = Graph.pmf_from_graph(g, use_cache=True) # Subsequent: instant from cache!`

Bit-identical across 16/16 `use_cache` pairs. Either wire it up or delete it and the
docstring.

### F6 — MEDIUM — `daisy_chain_t_eval` is a confirmed no-op under the DEFAULT `final_read='sojourn'`

This confirms §10 of the hand-off, by execution:

```
final_read='sojourn'                       final_read='stopprob'
  t_eval=0.05   -> [0.06772575, ...]         t_eval=0.05   -> [0.04439904, ...]
  t_eval=1.0    -> [0.06772575, ...]         t_eval=1.0    -> [0.06051579, ...]
  t_eval=10.0   -> [0.06772575, ...]         t_eval=10.0   -> [0.06772569, ...]
  t_eval=1000.0 -> [0.06772575, ...]         t_eval=1000.0 -> [0.06772575, ...]
  t_eval=None   -> [0.06772575, ...]         t_eval=None   -> [0.06772569, ...]
  ^ completely inert                         ^ live
```

Since `final_read='sojourn'` is the **default**, the default configuration silently ignores
`t_eval` — and `svgd()` still runs the expensive `_resolve_daisy_chain_t_eval` probe and
discards the result. Raise, or document.

*Bonus invariant (worth keeping as a test):* `stopprob` converges to the `sojourn` value as
`t_eval → ∞` (`0.06772575`), which independently corroborates that the granularity-free
sojourn read is the exact `t → ∞` limit of the legacy forward solve. Also note `t_eval=None`
under `stopprob` resolves to the same value as `t_eval=10.0` — an undocumented silent
default.

### F7 — LOW — an all-zero-state vertex makes `moments_from_graph` return `inf` / "NaN at vertex 0"

Found by accident (my first harness model tripped it). A **non-start** vertex whose state is
the all-zero vector (the same state the starting vertex carries) is accepted by
`find_or_create_vertex` as a distinct vertex, and `Graph.moments()` handles it correctly —
but `moments_from_graph` returns `[inf]` for `nr_moments=1` and raises
`XlaRuntimeError: Computation produced NaN at vertex 0` for `nr_moments ≥ 2`.

Identical on both sides → **not a regression**. Worth a guard or a doc note.

---

## 5. Explicitly cleared — do NOT chase these in later phases

The inert-flag sweep flagged these; each turned out to be sound. Recording them so Phase 2/3
does not re-investigate:

* **`reward_visit_probability` looks weight_mode-inert — it is not.** That was an artifact of
  my test model: on a *forced* chain the visit probability is topologically 0 or 1, so it
  cannot depend on rates. On a **branching** graph it is correct:
  `rvp(theta=[1,1]) = 0.5`, `[3,1] = 0.75`, `[1,3] = 0.25` — exactly `t0/(t0+t1)`.
* **`fixed_mask` is forward-inert — that is CORRECT.** It only skips positions in the FD
  backward. Forward-inertness is the expected, desired behaviour.
* **`theta_dim=None` vs `2` is inert — expected.** `2` *is* the inferred `param_length` for
  the test graph, so it is a no-op by construction. (`theta_dim=0` is the one value with a
  real effect — `_graph_serialize.py:76`.)
* **`joint_prob_graph(discrete=…)` produces a byte-identical graph — legitimate.**
  `serialize()` sha1 is the same for both (`6b4a1099d5750588`), but the `is_discrete`
  metadata flag does differ (`False`/`True`). Same graph, different downstream
  interpretation. My structural fingerprint simply did not capture metadata.

---

## 6. Coverage gaps — what Phase 1 did NOT establish

Stated plainly, because a partial sweep that reads as exhaustive is worse than no sweep.

1. **`exposure` / `exposure_param_index` were not exercised.** They are `svgd()`-level, and
   the original failure in §10 of the hand-off involved dropping them. **Phase 2/3 should
   cover them.**
2. **`epoch_starts` was not driven through `svgd()`.** The underlying epoch machinery *is*
   covered via `daisy_chain_joint_probs` (16 cells, 2 epoch counts), but not the
   `svgd(epoch_starts=…)` dispatch that builds it.
3. **Several `svgd()` cells raised on both sides due to harness argument forms**, not
   library defects: `discrete=True` with my continuous observations, and the `fixed=` /
   `tied=` forms I passed (`SvgdConfigError`). Identical on both sides, so parity is
   unaffected, but those flags are **not** covered. The correct argument forms need to be
   read out of `svgd_config.py`.
4. **One small model.** A 3-state chain (plus a 40-vertex coalescent for the joint/daisy
   paths). Bit-parity on a small graph is strong evidence for a change that is structurally
   forward-neutral, but it does not exercise SCC decomposition, MPFR, or the
   distributed/hierarchical caches.
5. **`granularity` on `pmf_from_graph` is not a user flag** — it is hardcoded to 0 (auto) at
   `:3612` / `:3697`; likewise `pmf_from_cpp` hardcodes 0 at `:4118` despite its inner
   function exposing it. So "granularity" was only swept where it is actually reachable
   (the daisy path), where it is live.
6. **Gradients were not audited** — that is Phase 2, by design. Note F1 hands Phase 2 a
   concrete, falsifiable lead.

---

## 7. Recommendation — and what was subsequently done

* The forward-parity claim is **sound**; the refactor does not regress any forward value.
* Carry the flag matrix in §8 into Phases 2–4. The §8 whitelist has been corrected to cover
  `moments_from_graph` in full (per **F4**).

### Batch A — implemented (see `audit-f1-f2-fix-plan.md`)

F1 and F2 were judged to block sign-off, and a minimal, raise-only Batch A was applied:

| | change | file |
|---|---|---|
| **A1** | `_fd_probe_points(theta, i, weight_mode)` — takes the mode, not a `nonneg` bool. `'log'` now uses a **sign-preserving multiplicative** probe (`θ·(1 ± 1e-6)`, no absolute floor) that cannot cross zero at any magnitude. `'linear'` keeps its strictly-positive floor; `callback`/`formula` keep no floor. | `__init__.py:702` + 12 call sites |
| **A2** | `moments_from_graph` **raises** for `weight_mode != 'linear'` instead of silently returning linear moments. Safe: it has **no internal callers** — `svgd()` never uses it. | `__init__.py` |
| **A3** | `pmf_from_graph_joint_index` and `daisy_chain_joint_probs` **raise** for `weight_mode='log'`. `callback`/`formula` are honoured and untouched. | `__init__.py` |
| **A4** | The false "θ is a log-scale" premise corrected in the code comments and in §3/§8 of the hand-off. | `__init__.py`, hand-off |

**Gate results:**

* Forward parity re-run vs HEAD: **167 bit-identical, 0 value differences, 0 *unintended*
  new raises.** The 4 cells that left the identical bucket are exactly the deliberate
  `joint_index|mode=log` raises (171 − 4 = 167).
* F1's hole is closed: `jax.grad` on a `log` graph is now finite at θ = 1e-15, 1e-16 and
  1e-20, where it previously raised.
* 21 new regression tests in `tests/pytest/test_weight_mode_probe_and_guards.py` — all pass.
  They assert **values against closed forms**, never `isfinite`/`> 0`, because the bugs they
  pin produced finite, plausible, wrong numbers.
* The `'linear'` FD branch is **bit-identical** to the old `nonneg=True` formula at all 10
  probe points (θ = 0 … 1e-30), so Batch A cannot alter a linear-graph gradient.
* 20-target regression subset (33 min): **580 passed, 50 skipped, 21 xfailed, 0 failed.**
  (`test_svgd_correctness::test_basic_convergence` is **flaky / order-dependent** — it failed
  once, then passed twice and passes in isolation. It uses a linear graph, so the bit-identity
  result above rules Batch A out as its cause. It is not in `failing_tests.md`.)

**Batch B (real support rather than guards) is deliberately NOT done** and does not gate
sign-off: thread `use_log` through the five FFI sites, route `moments_from_graph` through
`GraphBuilder`, and resolve the dead flags F4/F5/F6.

---

## 8. Flag matrix

Two matrices follow. The first is **empirical** — derived by executing every cell — and is
what later phases should reuse for cell selection. The second is **source-derived**, with
`file:line` evidence for every claim.

### 8.1 Empirical: flags actually swept, and whether each is live

`INERT` = toggling it never changed any forward value across otherwise-identical cells.

| entry point | flag | values swept | live? |
|---|---|---|---|
| `pmf_from_graph` | `discrete` | False, True | **live** (16/16) |
| | `weight_mode` | linear, log, callback, formula | **live** (12/24 — log differs; callback/formula ≡ linear, all being dot products) |
| | `use_cache` | True, False | `INERT` (16 pairs) → **F5** |
| | `theta_dim` | None, 2 | `INERT` (2 == inferred; expected) |
| `moments_from_graph` | `nr_moments` | 1, 2, 4 | **live** (16/16) |
| | `use_ffi` | False, True | `INERT` (12 pairs) → **F4** |
| | `weight_mode` | linear, log, callback, formula | `INERT` (18 pairs) → **F2 (wrong answers)** |
| `pmf_and_moments_from_graph` | `discrete` | False, True | **live** (32/32) |
| | `weight_mode` | 4 modes | **live** (24/48) |
| | `use_ffi` | False, True | **live** (2/32) |
| | `rewards` | absent, ones | **live** (2/32) |
| | `fixed_mask` | None, [1,0] | `INERT` — **correct** (backward-only) |
| `..._multivariate` | `discrete`, `weight_mode`, `use_ffi` | as above | **live** |
| `pmf_from_graph_joint_index` | `weight_mode` | 4 modes | `INERT` (12 pairs) → **F3 (wrong answers)** |
| | `theta_dim`, `fixed_mask` | None/2, None/[1,0] | `INERT` — expected |
| `pmf_from_cpp` | `discrete` | False, True | **live** |
| `daisy_chain_joint_probs` | `final_read` | sojourn, stopprob | **live** (8/8) |
| | `granularity` | 0 (auto), 100 | **live** (8/8) |
| | `n_epochs` | 1, 2 | **live** (8/8) |
| | `t_eval` | 0.05, 1, 10, 1000, None | `INERT` under `sojourn` (the **default**); **live** under `stopprob` → **F6** |
| `reward_visit_probability` | `rewards` | full, partial, all-zero | **live** (8/8) |
| | `weight_mode` | 4 modes | inert *on a forced chain only* — **not a defect** (§5) |
| `joint_prob_graph` | `discrete` | False, True | graph byte-identical; `is_discrete` metadata differs — **not a defect** (§5) |
| `SVGD.log_likelihood` | `discrete`, `weight_mode`, `rewards`, sparse/dense | see §2 | **live**; sparse + zero-inflated branches both exercised |

### 8.2 Silent overrides — the correctness traps, ranked

| # | trap | evidence | severity |
|---|---|---|---|
| 1 | `moments_from_graph` ignores `log`+`callback`+`formula` → **silently wrong moments** | F2; returns 0.325 where truth is 0.75 (callback/formula) / 0.333 (log) | **HIGH** — newly reachable; `formula` is mainstream |
| 2 | `joint_index` + both daisy handlers ignore `log` (hardcoded `use_log=false`) | F3; `graph_builder_ffi.cpp:887,941,1528,1782,1827` | MED — `log` unusable on zero-coefficient graphs anyway |
| 3 | `daisy_chain_t_eval` inert under default `final_read='sojourn'` | F6; `__init__.py:9390` default | MED |
| 4 | `moments_from_graph.use_ffi` selects nothing | F4; `__init__.py:6441`, `:6451` | MED |
| 5 | `pmf_from_graph.use_cache` never read; docstring promises caching | F5; `__init__.py:3478`, `:3501`, `:3551` | MED |
| 6 | `t_eval=None` silently resolves to the `10.0` value under `stopprob` | F6 | LOW |
| 7 | `pmf_from_cpp` granularity unreachable (hardcoded 0 at `:4118`) | source-derived matrix §2.3 | LOW |
| 8 | all-zero-state vertex → `inf`/NaN from `moments_from_graph` | F7 | LOW |

### 8.3 Source-derived flag matrix (full, with `file:line`)

The complete source-derived matrix — every entry point's signature, flag semantics, raise
guards, FFI-handler registry and orphan analysis — is reproduced below as generated from
the source by six independent readers.

<!-- BEGIN source-derived flag matrix -->
Merged from six source-reading passes (4 enumerators, 1 silent-override hunter, 1 raise-guard enumerator). Every claim below carries its originating `file:line`. Where readers disagree, the disagreement is called out inline and in the **Contradictions** section and marked **UNRESOLVED**.

All paths are relative to `/Users/kmt/phasic` unless already absolute.

---

## 1. ENTRY-POINT TABLE

| # | Entry point | Defined at | Signature (abbrev.) | Returned callable takes |
|---|---|---|---|---|
| 1 | `Graph.pmf_from_graph` | `src/phasic/__init__.py:3478` | `(cls, graph, discrete=False, use_cache=True, theta_dim=None)` | param graph → `model(theta, times)`; non-param → `model(times)` (one arg) |
| 2 | `Graph.pmf_from_graph_parameterized` | `src/phasic/__init__.py:3807` | `(cls, graph_builder, discrete=False)` | `model(theta, times, granularity=100)` — **broken, always raises** |
| 3 | `Graph.pmf_from_cpp` | `src/phasic/__init__.py:3967` | `(cls, cpp_file, discrete=False)` | `model(theta, times)` |
| 4 | `Graph.reward_visit_probability` | `src/phasic/__init__.py:3258` | `(self, rewards, theta=None)` | n/a — returns a scalar directly |
| 5 | `Graph.moments_from_graph` | `src/phasic/__init__.py:6382` | `(cls, graph, nr_moments=2, use_ffi=False)` | `moments_fn(theta)` → `(nr_moments,)` |
| 6 | `Graph.pmf_and_moments_from_graph` | `src/phasic/__init__.py:6594` | `(cls, graph, nr_moments=2, discrete=False, use_ffi=False, theta_dim=None, fixed_mask=None)` | `model(theta, times, rewards=None)` → `(pmf, moments)` |
| 7 | `Graph.pmf_and_moments_from_graph_multivariate` | `src/phasic/__init__.py:7589` | same 6 flags as #6 | `model(theta, times, rewards=None)` → `(pmf, moments)` |
| 8 | `Graph.pmf_from_graph_joint_index` | `src/phasic/__init__.py:7181` | `(cls, graph, theta_dim=None, fixed_mask=None, exclude_vertices=None, observed_indices=None)` | `model(theta, vertex_indices, rewards=None)` → `(probs, zeros(2))` |
| 9 | `Graph.joint_prob_graph` | `src/phasic/__init__.py:8312` | `(self, base_graph_indexer=None, reward_only=None, reward_rates_callback=None, mutation_rate=1.0, reward_limit=None, tot_reward_limit=inf, discrete=True)` | returns a `Graph` |
| 10 | `Graph.joint_stop_prob_graph` | `src/phasic/__init__.py:8989` | `(self)` — no args | returns a `Graph` |
| 11 | `Graph.joint_sojourn_graph` | `src/phasic/__init__.py:9177` | `(self)` — no args | returns a `Graph` |
| 12 | `Graph.daisy_chain_joint_probs` | `src/phasic/__init__.py:9390` | keyword-only: `(*, epoch_thetas, epoch_dts, initial_ipv, t_eval=None, fixed_indices=None, granularity=0, final_read='sojourn')` | returns an array directly |
| 13 | `Graph.svgd` | `src/phasic/__init__.py:5017` | 40 kwargs (see §7) | returns an `SVGD` object |
| 14 | `Graph.pdf / cdf / moments` | `src/phasic/__init__.py:2268 / 2310 / 2061` | `pdf(time, granularity=0)`, `cdf(time)`, `moments(power, rewards=[], discrete=False)` | scalar/array directly (no JAX) |
| 15 | `Graph.update_weights` | `src/phasic/__init__.py:1874` | `(self, theta, callback=None, log=False, weight_formula=None)` | mutates in place, returns `None` |
| 16 | `SVGD.__init__` | `src/phasic/svgd.py:4801` | `(model, observed_data, …, _validated=False)` | constructs the optimizer object |
| 17 | `Graph._daisy_chain_svgd_model` | `src/phasic/__init__.py:4164` | keyword-only builder (internal, driven by #13) | `model(theta_flat, _obs=None, rewards=None)` → `(probs, zeros(2))` |
| — | FFI wrappers (§6) | `src/phasic/ffi_wrappers.py` | `compute_pmf_ffi:523`, `compute_pmf_multivariate_ffi:781`, `compute_sojourn_times_ffi:936`, `compute_daisy_chain_joint_probs_ffi:1361`, `compute_daisy_chain_sojourn_ffi:1447` | bare `jax.ffi.ffi_call`s |

---

## 2. FLAG MATRIX (per entry point)

Legend: **V** = valid combination, **R** = raises, **S** = silent override (details in §3).

### 1. `Graph.pmf_from_graph` (`:3478`)

**Dispatch keys:** `graph.parameterized()` (`:3575`) then `serialized['weight_mode']` (`:3589`).

| Flag | Values | Effect / evidence |
|---|---|---|
| `graph` | any | selects param vs non-param branch (`:3575`) |
| weight_mode (off graph) | linear/log/formula → FFI branch (`:3660`); callback → pure_callback branch (`:3589`); C++ throws `Unknown weight_mode` for anything else (`graph_builder.cpp:29-36`) |
| `discrete` | False/True | static attr to all branches (`:3612,3696,3793`); C++ int-casts `times` when True (`graph_builder_ffi.cpp:192-201`) |
| `theta_dim` | None/int | only `graph.serialize(theta_dim=…)` (`:3568`); `theta_dim=0` is the ONLY value that makes a constant graph serialize edges as regular edges (`_graph_serialize.py:76`) |
| `use_ffi` (config only) | `config._use_ffi` (`config.py:603`), read at `:3672` | not a call arg |
| `granularity` | hardcoded 0 (`:3612,3697`) → auto in C (`src/c/phasic.c:11380-11405`) |
| `use_cache` | **accepted, never read** (`:3478` sig only) |

**V:** param + linear/log/formula + ffi on; param + callback + `weight_callback` set; non-param + **`theta_dim=0`** → `model(times)`.
**R:** JAX missing → `ImportError` (`:3555-3560`); callback mode + `weight_callback is None` → `ValueError` (`:3596-3600`); param + `configure(ffi=False)` → `PTDBackendError` (`:3674-3686`); zero non-start vertices → `ValueError` from serialize (`_graph_serialize.py:42-43`); `discrete=True` called with integer `times` → `XlaRuntimeError` "Wrong buffer dtype: expected F64 but got S64" (`ffi_wrappers.py:588-599`; `graph_builder_ffi.cpp:95-96`).
**S:** see §3 #1 (non-param → zeros), #6 (`use_cache`), #14 (granularity), FD gradients not autodiff, arity changes with graph.

### 2. `Graph.pmf_from_graph_parameterized` (`:3807`) — **DEAD / BROKEN**

`graph_builder` is called as `graph_builder(*theta_np)` — theta splatted positionally (`:520`). No valid combination.
**R (both are alternatives, same failing line `:535`):**
- Fresh process, JAX never activated in-proc: `AttributeError: 'NoneType' object has no attribute 'ShapeDtypeStruct'` (`:535`; module globals `jax=None/jnp=None` at `:143-144`, never populated because the method never calls `_ensure_jax_active`).
- JAX already activated: `XlaRuntimeError … RuntimeError: Incorrect output dtype … Expected: float32, Actual: float64` — `:535` declares `jnp.float32`, ctypes returns float64 (`:432/:496`), x64 forced (`:125,263`).

One reader (pass 4) observed only the dtype variant; pass 1 distinguished both by process state. Not contradictory — pass 1 is a superset.
**S:** granularity default 100 differs from all others (`:509`); lib cache keyed on `inspect.getsource` SHA-256 (`:3944-3948`) → source-text collisions; no `custom_vjp` (`:501-538`).

### 3. `Graph.pmf_from_cpp` (`:3967`)

| Flag | Effect |
|---|---|
| `cpp_file` | resolved/existence-checked/substring-checked for `build_model` (`:4021-4049`) |
| `discrete` | `compute_dph_pmf` vs `compute_pmf` (`:4099`); discrete wrapper does NOT `normalize()` (`:4066-4079`) — deliberately unlike #2 |
| granularity | inner fn exposes it (`:4102`) but jax wrapper hardcodes 0 (`:4118`) |

**V:** discrete=False `model(theta,times)`→f64; discrete=True with int-castable times; `jax.grad/jit` (central FD, `nonneg=False`, `:4126-4160`).
**R:** missing file → `FileNotFoundError` before JAX check (`:4022-4023`); JAX missing → `ImportError` (`:4026-4031`); no `build_model` substring → `ValueError` (`:4038-4041`); no source tree + `PHASIC_SOURCE_DIR` unset → `RuntimeError` (`:1003-1016`); compile/link failure → `RuntimeError` (`:1070-1076`).
**S:** granularity unreachable (`:4118`); cache key = user file content only, not the wrapper template (`:4084-4096`); dtype hard-forced f64 (`:4116` — this is *why* #3 works and #2 doesn't); no `_source_graph` attr (`:4160-4161`).

### 4. `Graph.reward_visit_probability` (`:3258`)

| Flag | Values | Effect |
|---|---|---|
| `rewards` | shape `(n_vertices,)` | only `rewards > 0.0` pattern used (`:3310`); magnitudes & negatives discarded |
| no positive entry | early return 0.0 (`:3311-3320`), graph untouched |
| `theta` | None/np → concrete path; jax.Array/Tracer → FFI path (`:3326`) |

**V:** ≥1 positive reward + theta=None (current weights); + np theta (**mutates graph**); + jax theta (jax scalar). All-zero rewards → 0.0.
**R:** `rewards.shape` mismatch → `ValueError` (`:3304-3308`); concrete theta not 1-D / empty / NaN / inf → `ValueError` from `update_weights` (`:1925-1932`, called at `:3345`); FFI path on callback-mode graph → `std::invalid_argument('Unknown weight_mode: callback')` (`graph_builder.cpp:35-36`, **not** end-to-end read-verified); JAX missing → **no raise**, both `import jax` blocks fall through to concrete path (`:3318-3319,3338-3339`).
**S:** see §3 #7 (permanent mutation, docstring lies), #8 (log/callback ignored on concrete path; jax vs np different numbers), magnitudes discarded, result not clipped to [0,1].

### 5. `Graph.moments_from_graph` (`:6382`)

| Flag | Effect |
|---|---|
| `nr_moments` | output length; `nr_moments<=0` unguarded |
| `use_ffi` | **dead** — its only use is `if not use_ffi: _ensure_jax_active()` (`:6441`); no FFI path exists; both values take the same JIT-C++/ctypes path |
| weight_mode | **ignored** for the forward result (linear codegen hardwired) except FD polarity at `:6569` |

**V:** parameterized + linear + `PHASIC_SOURCE_DIR` set. Returns RAW moments `E[T^k]` (k! applied in generated C++ at `:6514`).
**R:** param_length==0 → `ValueError` (`:6457-6461`); JAX missing → `ImportError` (`:6441-6448`); no source tree → `RuntimeError` (`:1003-1016`); theta→zero exit rate or all-zero absorbing state → `XlaRuntimeError` NaN catastrophe (`src/c/phasic.c:12504`; aliasing at `:868-877`).
**S:** see §3 #2 (weight_mode ignored → wrong numbers for log/callback/formula), `use_ffi` dead, no `theta_dim`/`fixed_mask`/`discrete`, recompiles `.so` every call (`:975-976`).

> Reader note: `tests/pytest/inference/test_jax_integration.py:386` calls `moments_fn()` with no args — that test cannot pass (signature is `moments_fn(theta)`).

### 6. `Graph.pmf_and_moments_from_graph` (`:6594`)

| Flag | Effect |
|---|---|
| `discrete` | pdf vs dph_pmf, all three branches (`:6719,6876,6920`) |
| `use_ffi` | False → pybind+pure_callback (exposes `_cdf_zero_fn`); True → re-read as `config._use_ffi` (`:6861`); FFI path does NOT attach `_cdf_zero_fn` (`:7169`) |
| `theta_dim` | forwarded to serialize (`:6680`); wrong value fails at CALL time (`src/c/phasic.c:4842`) |
| `fixed_mask` | backward-only FD skip (`:6692,7130-7132`) |
| weight_mode='callback' | separate 3rd path (`:6700`); forces `use_ffi=False` (`:6848`) |
| granularity | pinned 0 (`:6720,6877,6922,6935`) |
| `rewards` (runtime) | None/1D/2D — 2D BROKEN on pybind path (see below) |

**V:** use_ffi=False + linear/log/formula + None/1D rewards (has `_cdf_zero_fn`); use_ffi=True + FFI available + 2D rewards `(n_vertices,n_features)`; callback + `weight_callback`; `discrete=True` requires genuine DPH; `nr_moments>=1`.
**R:** no param edges → `ValueError` (`:6683-6687`); JAX missing → `ImportError` (`:6666-6674`); callback + no `weight_callback` → `ValueError` (`:6706-6709`); log-mode + non-positive `coeff*theta` → `RuntimeError`/`XlaRuntimeError` (`graph_builder.cpp:455`); `weight_callback` returns NaN/inf → `ValueError` (`:790-801`); **2D rewards on default pybind path → `RuntimeError` "Incorrect output shape"** (`:6949` reads `shape[1]` vs `graph_builder.cpp:666` reads `shape[0]`); **`nr_moments=0` → SEGFAULT (exit 139)**, no guard (`graph_builder.cpp:480-491`); `discrete=True` on start out-rate>1 → `RuntimeError`.
**S:** see §3 #13 (use_ffi downgrade / callback force / `_cdf_zero_fn` drop), #22 (2D orientation inconsistency), granularity pinned, `nansum` for pmf vs `sum` for moments in backward (`:7144-7147`), `model_bwd` returns None grads for times/rewards (`:7151`). **Observed anomaly (not root-caused):** on a 50/50 mixture `_cdf_zero_fn` returns 1.0 where true atom mass is 0.5 (`graph_builder.cpp:923`, granularity hard-0 from `:7029`).

### 7. `Graph.pmf_and_moments_from_graph_multivariate` (`:7589`)

All 5 flags forwarded VERBATIM to #6 (`:7681-7684`) — inherits every semantic incl. the use_ffi downgrade, callback branch, log raise, and `nr_moments=0` segfault.

| Runtime flag | Effect |
|---|---|
| `rewards` | None/1D → straight to `model_1d`; 2D `(n_features,n_vertices)` → Python for-loop over rows (`:7743-7757`) so the broken 2D pybind path is never hit; ndim>2 → `ValueError` (`:7766-7769`) |
| `times` | 1D broadcast to every feature (`:7752`); 2D column-per-feature; `SparseObservations` → per-feature slices, empty feature → NaN moments (`:7719-7724`) |

**S:** see §3 #22 — reward orientation here is `(n_features,n_vertices)`, the **transpose** of the FFI wrapper's `(n_vertices,n_features)` (`ffi_wrappers.py:749-753`); plain python fn, not `custom_vjp` (`:7686`); forwards `_cdf_zero_fn` only if present (`:7776-7777`) so use_ffi=True silently loses zero-inflation.

### 8. `Graph.pmf_from_graph_joint_index` (`:7181`)

| Flag | Effect |
|---|---|
| `observed_indices` | MAIN switch (`:7304`). Not-None → bakes unique indices, **runtime `vertex_indices` ignored** (param literally `_vertex_indices_ignored`, `:7368,7470`); wraps FFI in `custom_vmap` |
| `exclude_vertices` | removes vertices from the **normalisation denominator only** (`:7289-7291`) — numerator untouched, probs can exceed unconditional value |
| `fixed_mask` | backward-only FD skip (`:7543-7546,7561-7563`) — built inline, not via `_fixed_indices_set_from_mask` |
| weight_mode='callback' | separate path (`:7325`), pybind `expected_sojourn_time`; no custom_vmap |
| `rewards` (runtime) | **accepted and never read** (`:7518,7535`) |
| discrete | **not a parameter** — reads exact sojourn times, not PDF/PMF |

**V:** any param graph + FFI + linear/log/formula → `model(theta, vertex_indices)`→`(probs, zeros(2))`; callback + `weight_callback`; `observed_indices` baked → output length = `len(observed_indices)`.
**R:** no param edges → `ValueError` (`:7271-7275`); JAX missing → `ImportError` (`:7256-7261`); callback + no callback → `ValueError` (`:7331-7337`); log + non-positive product → `RuntimeError` (`graph_builder.cpp:455`); FFI disabled → `PTDConfigError`/`PTDBackendError` (`ffi_wrappers.py:161-186,960-967`); passing `rewards` → **NOTHING raised here** (only `Graph.svgd` raises `NotImplementedError` at `:5760-5764`).
**S:** see §3 #16 (baked indices ignore runtime arg), #17 (dummy `zeros(2)` moments; `_precondition_output='probability'` at `:7585`), rewards ignored, exclude affects denominator only, `model_bwd` None grads for vertex_indices/rewards (`:7576`), terminal detection structural (`:7280-7292`).

### 9. `Graph.joint_prob_graph` (`:8312`)

| Flag | Effect |
|---|---|
| `discrete` | ONLY sets `joint_graph.is_discrete` + `set_was_dph(discrete)` (`:8691`); topology identical for both (nv=25 verified either way) |
| `reward_rates_callback` | **BROKEN** — any non-None value calls `self._joint_prob_graph(...)` which does NOT exist (delegate commented out `:8704-8986`) → `AttributeError` (`:8380-8389`) |
| `reward_limit` | int → per-dim cap (`:8467`); None → no cap; dict → documented-dead branch (`:8489-8495`), treat as unsupported |
| `tot_reward_limit` | total cap (`:8501`) |
| `reward_only` | restricts reward slots (`:8398`); listing all props silently reset to None (`:8638-8644`) |
| `mutation_rate` | baked into edge coefficient (`:8486,8620`) |
| `base_graph_indexer` | None+no `_indexer` → `TypeError` (`:8348`); state_length mismatch silently swapped for `self._indexer` (`:8355-8363`) else `ValueError` (`:8365`) |

**V:** callback=None AND (reward_limit set OR tot_reward_limit finite) AND param_length>0 AND indexer has exactly 1 property set.
**R:** param_length==0 → `ValueError` (`:8339-8340`); both limits at default → `ValueError` (`:8341-8342`); no indexer → `TypeError` (`:8348`); state_length mismatch unrescuable → `ValueError` (`:8365-8370`); ≠1 property set → `ValueError` (`:8372-8373`); reward_rates_callback set → `AttributeError` (`:8381`).
**S:** see §3 #3 (**UNRESOLVED** weight_mode → linear), #4 (`is_discrete` not serialized), indexer swap silent, reward_only reset silent, joint theta dim forced `param_length+1` (`:8530-8538`). Pass-4 also flags a possible loop-variable leak: `mask` read at `:8670` outside the `for` that defines it (`:8663-8664`).

### 10. `Graph.joint_stop_prob_graph` (`:8989`) — no args

**V:** only on a `joint_prob_graph()` result with `_joint_prob_base_graph_indexer`, param_length>0, exactly 2 trash + 1 absorbing vertex.
**R:** not a joint graph → `ValueError` (`:9029-9033`); param_length==0 → `ValueError` (`:9034-9038`); ≠2 trash vertices → `ValueError` (`:9069-9073`); no absorbing → `ValueError` (`:9074-9077`).
**S:** t-vertices lose original edges, get constant weight-1.0 aux trap (`:9102-9119`); edges to trash redirected to absorbing (`:9097-9098,9122-9124`); IPV edges weight 0.0, no guard before read (`:9147-9148`); `is_discrete` copied (`:9166`) but not serialized → computed continuous. **Contradiction on weight_mode** — see §3 #3.

### 11. `Graph.joint_sojourn_graph` (`:9177`) — no args

**V:** same preconditions as #10. Auto-called by `daisy_chain_joint_probs(final_read='sojourn')` (`:9557`) and `_daisy_chain_svgd_model` (`:4502,4706`).
**R:** not a joint graph → `ValueError` (`:9214-9217`); param_length==0 → `ValueError` (`:9218-9222`); ≠2 trash → `ValueError` (`:9251-9255`); no absorbing → `ValueError` (`:9256-9259`).
**S:** `new.is_discrete = False` **hard-forced** regardless of source (`:9313-9315`); trash-pair vertices NOT created here → different vertex count/indexing from #10 (23 vs 34 in tiny model), cross-graph mapping by state tuple (`:9268-9271,9558-9579`); IPV weight-0 placeholders (`:9304-9305`). Pass-4: `_cache_trace` NOT copied here though #10 copies it (`:9308-9317` vs `:9167`). C++ read at `graph_builder_ffi.cpp:1844-1866`.

### 12. `Graph.daisy_chain_joint_probs` (`:9390`) — keyword-only

| Flag | Effect |
|---|---|
| `final_read` | 'sojourn' (default) → `compute_daisy_chain_sojourn_ffi` (`:9607`), exact elimination final read (`graph_builder_ffi.cpp:1846-1865`); 'stopprob' → `compute_daisy_chain_joint_probs_ffi` (`:9613`), `stop_probability(t_eval)` final (`cpp:1569`) |
| `t_eval` | None → `max(sum(epoch_dts)*4, 10.0)` (`:9506-9507`); **NO-OP under 'sojourn'** — `DaisyChainSojournFfiImpl` never reads it (`cpp:1699-1707` vs `1328/1569`) |
| `granularity` | 0=auto; used by intermediate-epoch `stop_probability` in both handlers (`cpp:1797`); NOT the final epoch under 'sojourn' |
| `fixed_indices` | backward-only FD skip (`:9629-9647`); no forward effect |
| `epoch_thetas`/`epoch_dts` | `n_epochs=shape[0]`; flat row-major (`:9656`); `epoch_dts` static |
| `is_discrete` | **not checked, not serialized** → ignored (`_graph_serialize.py:201-217`) |

**V:** self is a `joint_stop_prob_graph()` result; 'sojourn' additionally needs `_joint_prob_source`; `len(epoch_dts)==n_epochs-1`; `epoch_thetas.shape[1]==param_length`; `initial_ipv.shape==(len(_ipv_target_indices),)`; 'stopprob'+finite t_eval is the only combo where t_eval matters.
**R:** not JSP graph → `ValueError` (`:9469-9473`); `epoch_thetas.ndim!=2` (`:9477`); `n_epochs<1` (`:9483`); theta_dim mismatch (`:9485`); epoch_dts length (`:9492`); ipv shape (`:9500`); `t_eval<=0` → `ValueError` **even under 'sojourn' where it is unused** (`:9508`); granularity not non-neg int (`:9510`); `final_read` invalid (`:9531`); 'sojourn' without `_joint_prob_source` (`:9550`); negative edge weight → `XLA InvalidArgument` (`graph_builder_ffi.cpp:1869-1878,1631-1640`).
**S:** see §3 #14 (t_eval no-op + expensive discarded auto-probe), #4 (is_discrete ignored — verified continuous numbers from a discrete graph), fixed_indices forward no-op, granularity partial, docstring describes stale 'stopprob' semantics (`:9415-9436`).

### 13. `Graph.svgd` (`:5017`) — see §7 for the full flag matrix (40 kwargs).

### 14. `Graph.pdf / cdf / moments` (`:2268 / 2310 / 2061`) — direct, non-JAX

| Flag | Effect |
|---|---|
| `self.is_discrete` (plain attr, no validation, `:1545`) | pdf→`pdf_discrete` vs `pdf(t,granularity)` (`:2305-2308`); cdf likewise (`:2331-2334`) |
| `moments(discrete=...)` | True→`moments_discrete`, requires `is_discrete` True |

**R:** `moments(discrete=True)` on continuous → `ValueError` (`:2110-2111`); NaN time (`:2296`); negative time (`:2298`); non-int granularity → `TypeError` (`:2300`); negative granularity (`:2302`); bad rewards (`:2145-2152`); `accumulated_visits()` on continuous (`:2561`); laplace on discrete (`:3222`); `is_discrete=True` + row sum >1.0001 → `RuntimeError` (`src/c/phasic.c:11182`).
**S:** `pdf(time, granularity=G)` **silently discards granularity when discrete** (validated then never forwarded, `:2300-2306`); `is_discrete` is a mutable attr, not serialized; `update_weights` on a `was_dph` graph **silently renormalises rows** (`src/c/phasic.c:5759-5781`) — opposite of the FFI path. `moments()` returns an array despite `-> float` annotation.

### 15. `Graph.update_weights` (`:1874`)

| Flag | Effect |
|---|---|
| `callback` | one-shot C++ overload (`:1950-1952`), does not change persistent mode |
| `weight_formula` | one-shot, temporarily flips mode, restored in finally (`:1939-1949`) |
| `log` | only reaches C++ in the plain branch (`:1954`) |

**R:** theta not 1-D (`:1925`); empty + no callback (`:1927`); NaN (`:1929`); inf (`:1931`); both callback and weight_formula (`:1933-1935`); linear/log len mismatch → C `ptd_err` (`src/c/phasic.c:5565-5572`).
**S:** `log=` **silently ignored** when `weight_formula=` given (literal `False` at `:1943`) or when `callback=` given (two-arg overload, `:1951-1952`); **negative weights unguarded** (C guard commented out `src/c/phasic.c:5745-5755`); `was_dph` auto-renormalises (`5759-5781`).

### 16 / 17. `SVGD.__init__` (`svgd.py:4801`) and `_daisy_chain_svgd_model` (`:4164`) — see §7.

---

## 3. SILENT OVERRIDES / IGNORED ARGUMENTS — CORRECTNESS TRAPS

Ranked most-dangerous first. **#1–#8 change returned numbers or persistent state; #9–#16 change behaviour/likelihood; #17+ are ignored/wasteful.**

1. **`pmf_from_graph` non-parameterized branch returns ALL ZEROS with default args.** `serialize(theta_dim=None)` → `theta_dim = param_length()` which the C layer sets to 1 even for a constant edge, so the edge is exported to `param_edges` (`_graph_serialize.py:76`) and codegen emits `w = 2.0*theta[0]` (`:932-937`) while `non_param_wrapper` feeds `dummy_theta=[0.0]` (`:3800`) → every rate 0, pdf 0 everywhere. **The library's own docstring example (`:3518-3528`) returns `[0.,0.]` instead of `[0.7365,0.2707]`.** Fix: pass `theta_dim=0`. Evidence: `:3568,3575,3800`; `_graph_serialize.py:64-65,76`; `:932-937`.

2. **`moments_from_graph` ignores `weight_mode` → silent WRONG numbers for log/callback/formula graphs.** Codegen is hardwired linear (`:920-923,932-935`, no `weight_mode` read). Verified: an exp-weight graph returns the LINEAR `[2.,6.]` while `pmf_and_moments_from_graph` returns the correct `[0.7358,0.8120]`; for `log` mode it returns linear while every other entry point RAISES. Evidence: `:837-954`.

3. **[UNRESOLVED] `weight_mode` propagation into joint-prob graphs.** Pass 4 verified by execution that a `_weight_mode='log'` base graph yields `jp._weight_mode == 'linear'` on `joint_prob_graph`, `joint_stop_prob_graph`, and `joint_sojourn_graph` — a log model silently becomes linear (`:1552` default; no assignment in `8312-8703`, `9161-9167`, `9308-9317`). Pass 3 claims linear/log "reach the FFI anyway through `serialize()['weight_mode']`" and that only `formula` needs explicit propagation (`_propagate_weight_formula`, `:9173/9319`; helper returns early for other modes `:614-615`). **These cannot both be true for the log case.** Whether log semantics survive into joint-prob inference is UNRESOLVED and must be settled by executing a log joint-prob model end-to-end.

4. **`is_discrete` is not serialized → discrete joint-prob graphs are computed as CONTINUOUS by every FFI/daisy path.** `serialize()` emits no `is_discrete`/`was_dph` key (`_graph_serialize.py:201-217`). Verified: a `joint_prob_graph(discrete=True)` run through `daisy_chain_joint_probs` returns bit-identical continuous numbers (sum `0.6373456790123462`) with no error. `daisy_chain_joint_probs` has no discrete guard (contrast `_daisy_chain_svgd_model:4267-4273` and `svgd_config.py:702-708` which DO raise). `joint_sojourn_graph` additionally hard-forces `is_discrete=False` (`:9313-9315`).

5. **Discrete row-sum: same model, two answers depending on entry point.** A live `was_dph` graph SILENTLY RENORMALISES rows to sum 1 inside `update_weights` (`src/c/phasic.c:5759-5781`), so the guard never fires; the FFI/JAX path rebuilds from `serialize()` (no `was_dph`), so the row-sum guard at `src/c/phasic.c:11182` RAISES. Guard tolerance is an **absolute** `rate > 1.0001` — a row sum in `(1.0, 1.0001]` passes and yields "probabilities" >1 (verified `theta=1.00005` → pmf `1.00005`). Evidence: `phasic.c:11172-11208`; `_graph_serialize.py:201-213`; `__init__.py:2730` (discretize sets `set_was_dph(True)`).

6. **Negative rates are UNGUARDED everywhere except the two daisy-chain FFI handlers.** The C negative-weight check in `ptd_graph_update_weights` is commented out (`src/c/phasic.c:5745-5755`); linear `compute_weight` has no sign check (`graph_builder.cpp:463-469`). Verified: `pmf_from_graph` with `theta=-1.0` returns pdf `-1.6467`, no error. Only `kMinEdgeWeight=-1e-12` in `graph_builder_ffi.cpp:1238` (daisy only) rejects it.

7. **`reward_visit_probability` concrete path PERMANENTLY MUTATES the graph.** Docstring says weights are "temporarily updated … then restored" (`:3342-3343`) but there is NO restore after `self.update_weights(theta)` at `:3345`. Verified: an edge weight of 2.0 reads 18.0 after `reward_visit_probability(r, theta=[9.0])`; every later `pdf/moments` uses the probe theta.

8. **SVGD `learning_rate` is scaled by `lr_scale` TWICE.** `lr_scale = 1/max(1, n_obs/1000)` applied at construction (`svgd.py:5173-5175`) and again in the loop (`svgd.py:4259-4260`) → effective step `learning_rate * lr_scale**2` for >1000 observations. Evidence: `svgd.py:5152,5173-5175,6647/6706,4214-4220`.

9. **SVGD `jit=False` is silently forced back to True by the default `precompile=True`.** `if precompile and not jit: self.jit_enabled = True` (`svgd.py:5001-5004`). To truly disable JIT you must pass BOTH `jit=False` and `precompile=False` (which then also emits a DeprecationWarning).

10. **SVGD fixed values ≤0 bypass inverse-softplus while softplus is still applied at eval.** `fixed=[(0, 0.0)]` with default `positive_params=True` makes the model see `softplus(0)=0.693`, not 0.0. Evidence: `svgd.py:5460-5466,3979-3984,5218-5225,6044-6052`.

11. **`validate_rewards=False` silently DISABLES the zero-inflated likelihood, not just validation.** The partial-coverage scan is inside the same guard (`:5635-5645`), so `_attach_zero_inflated_term` is never called (`:5917`) — the posterior optimises a different likelihood.

12. **`use_ffi=True` on `pmf_and_moments_from_graph` is silently downgraded to `config._use_ffi`** (`:6858-6861`) and hard-cleared to False in callback mode (`:6848`); it also silently DROPS `model._cdf_zero_fn` (`:7169`). Unlike `pmf_from_graph`, which RAISES `PTDBackendError` in the FFI-disabled case. SVGD always runs the pybind path (it passes/leaves `use_ffi=False`, `:5892-5896`).

13. **`reward_visit_probability` concrete path silently ignores `log`/`callback` weight_mode; jax vs np theta take different code paths with different numbers.** `update_weights` called with defaults `callback=None, log=False` (`:3345`; dispatch `:1953-1955`) so a `log`-mode graph computes as linear. A concrete `jnp.array` is caught by the `isinstance` at `:3326` and routed through the FFI (which honours log/formula) — so `jnp.array([x])` and `np.array([x])` give different results and different return types.

14. **`daisy_chain_t_eval` (incl. the expensive `'auto'` probe), `daisy_chain_probe_theta`, `daisy_chain_t_eval_tol` are IGNORED under the default `final_read='sojourn'`.** `t_eval` is validated, defaulted, and written into JSON but `DaisyChainSojournFfiImpl` never reads it (`graph_builder_ffi.cpp:1699-1707` vs `1328/1386`). `Graph.svgd` calls `_resolve_daisy_chain_t_eval` UNCONDITIONALLY (`:5806`) so the auto-probe runs (`:9859`) and its result is discarded. Verified: t_eval=0.001 vs 200.0 → bit-identical output.

15. **`joint_index` is unconditionally overwritten to True on a joint graph** (`:5729`, source FIXME admits it), and **`discrete` is forced True (`:5778`) but never read again** on that path — the joint/daisy builders take no `discrete` argument. The `DataPrior` built at `:5696` consumed the PRE-forced value. On the standard/reward/multivariate paths, `final_read` is accepted and ignored — including its own validity check (`Graph.svgd(data, final_read='garbage')` on a standard graph runs happily; validation only inside the daisy builder at `:4448`).

16. **`pmf_from_graph_joint_index` returns dummy `jnp.zeros(2)` as its "moments".** Any moment regularization/moment-Jacobian preconditioning silently degenerates; `_precondition_output='probability'` is stamped to compensate (`:7513,7581-7585`). The runtime `rewards` arg is accepted and never read (`:7518`).

17. **`use_cache` on `pmf_from_graph` is accepted and NEVER read** (`:3478` sig, `:3501/3551/3552` docstring/doctest only; body says the symbolic cache was removed `:3562-3565`). No-op.

18. **`pmf_from_graph_parameterized` is a dead entry point** (dtype/activation crash at `:535`) — see §2 #2.

19. **granularity is hard-forced to 0** in `pmf_from_graph` (`:3612,3697`), `pmf_and_moments_from_graph` (`:6720,6877,6922,6935`), `pmf_from_cpp` jax wrapper (`:4118`) — not exposed. `pmf_from_graph_parameterized` uses 100 (`:509`), inconsistent.

20. **`discrete=True` silently TRUNCATES `times` to int** (`static_cast<int>`, `graph_builder_ffi.cpp:193-195`) — `times=2.7` → jump 2, no warning.

21. **2D-reward orientation is inconsistent across paths.** pybind reads `shape[1]` (`:6949`), C++ reads `shape[0]` (`graph_builder.cpp:666`), callback reads `shape[0]` (`:6727`), FFI wrapper reads `shape[1]` (`ffi_wrappers.py:749-753`), the multivariate wrapper expects `(n_features,n_vertices)` (`:7714,7745`) and `compute_pmf_multivariate_ffi` silently transposes with `swapaxes` (`ffi_wrappers.py:901-908`). Net: default path can never do 2D rewards; callback vs FFI expect transposed matrices for the same builder.

22. **`compute_sojourn_times_ffi` float indices silently truncated** — the int32 check at `ffi_wrappers.py:1016` is unreachable because `:1014` already casts (`jnp.asarray(..., dtype=jnp.int32)`).

23. **SVGD `result.theta_mean` / `theta_std` are in UNCONSTRAINED (phi) space**, not the constrained theta the model saw. With default `positive_params=True` they can be negative (verified `-0.889` for a rate-2 model). Evidence: `svgd.py:4296-4297,6712-6714`.

24. **Multi-index joint observations resolved with UNSEEDED `np.random.choice`** (`:5742-5747`) — `Graph.svgd`'s `seed` does not make the joint observation mapping reproducible.

25. **`_apply_weight_callback` rewrites the serialized dict** to `weight_mode='linear'`, `param_length=0` before the callback branch (`:826-832`) — so log/formula/negative C++ guards never fire in callback mode.

26. **The 'All parameters are fixed!' guard is nested inside `if verbose:`** in both branches (`svgd.py:5470-5476,5509-5513`) → with default `verbose=False` an all-fixed config is accepted silently.

27. **`pmf_from_graph` writes the generated `.cpp` to disk on EVERY call** (`:3578-3586`) even on FFI/callback branches where it is unused. Pure waste.

28. **`fixed_mask_for_model` (`:5688`) is dead on both joint branches** — recomputed at `:5862-5870` (non-daisy joint) or ignored (daisy). Harmless today but built from the pre-daisy `theta_dim`.

29. **`n_devices` silently reset to None when `parallel != 'pmap'`; `parallel='pmap'` on 1 device silently downgraded to 'vmap'** (`svgd.py:4947-4976`). Models tagged `_handles_exposure_internally` without `_handles_particle_vmap` have `parallel` forced to 'none' with no warning (`svgd.py:5698-5708`; dormant today).

30. **Gradients on ALL of #1/#3/#5/#6/#7/#8/#12/#17 are finite differences, not autodiff**, despite docstrings advertising "full gradient support." `fd_nonneg` (positivity floor) is applied only when `weight_mode=='linear'` (`:3762-3764, 6569, 6697, 7548, 9623`).

---

## 4. FFI HANDLER REGISTRY & ORPHANS

Chain: C++ impl (`src/cpp/parameterized/graph_builder_ffi.cpp`) → `Create*Handler` → XLA target name (registered in `src/phasic/ffi_wrappers.py`) → Python wrapper → in-`src` caller.

| # | C++ impl | Handler | XLA target | Python wrapper | In-`src` caller |
|---|---|---|---|---|---|
| 1 | `ComputePmfFfiImpl` (`ffi.cpp:91`) | `1892` | `ptd_compute_pmf` (`ffi_wrappers.py:223`) | `compute_pmf_ffi:523` | `__init__.py:3694` |
| 2 | `ComputeMomentsFfiImpl` (`229`) | `1909` | `ptd_compute_moments` (`229`) | `compute_moments_ffi:603` | **ORPHAN** — re-exported at `__init__.py:396`, never called in `src`; tests only (`test_gate_moments_3way.py:80`, `test_ffi_multi_process.py:130`) |
| 3 | `ComputePmfAndMomentsFfiImpl` (`325`) | `1924` | `ptd_compute_pmf_and_moments` (`235`) | `compute_pmf_and_moments_ffi:677` | `__init__.py:6873` |
| 4 | `ComputePmfMultivariateFfiImpl` (`519`) | `1944` | `ptd_compute_pmf_multivariate` (`241`) | `compute_pmf_multivariate_ffi:781` | **ORPHAN** — tests only (`test_reward_validation.py:417`) |
| 5 | `ComputeSojournTimesFfiImpl` (`765`) | `1963` | `ptd_compute_sojourn_times` (`257`) | `compute_sojourn_times_ffi:936` | `__init__.py:7436/7448/7456/7464/7500/7504`, `bffg.py:418/421/581/584` |
| 6 | `BackwardProbabilitiesFfiImpl` (`984`) | `1978` | `ptd_backward_probabilities` (`267`) | `backward_probabilities_ffi:1118` | **DEAD** — no caller anywhere in `src/` or `tests/`; the reward-visit path uses the underlying C handler, not this wrapper (comment `ffi_wrappers.py:1173-1175`) |
| 7 | `SamplePathConditionedFfiImpl` (`1090`) | `1992` | `ptd_sample_path_conditioned` (`275`) | `sample_path_conditioned_ffi:1063` | `bffg.py:532` |
| 8 | `DaisyChainJointProbsFfiImpl` (`1271`) | `2009` | `ptd_daisy_chain_joint_probs` (`288`) | `compute_daisy_chain_joint_probs_ffi:1361` | `__init__.py:4554,4769,9613` (`final_read='stopprob'`) |
| 9 | `DaisyChainSojournFfiImpl` (`1663`) | `2023` | `ptd_daisy_chain_sojourn` (`300`) | `compute_daisy_chain_sojourn_ffi:1447` | `__init__.py:4551,4766,9607` (`final_read='sojourn'`, the DEFAULT) |

**Not an XLA FFI handler:** `compute_reward_visit_probability_ffi` (`ffi_wrappers.py:1162`) routes through `jax.pure_callback` + the pybind builder; caller `__init__.py:3332`.

**Orphan summary:** #2 and #4 are reachable only from tests. **#6 `BackwardProbabilitiesFfiImpl` is fully dead** (registered, never called). Registration of the BFFG, daisy-joint-probs and daisy-sojourn targets is wrapped in `except AttributeError: pass` (`ffi_wrappers.py:262-306`) — a build missing those handlers fails LATE with an opaque XLA "target not found" rather than `PTDBackendError`.

FFI-wrapper flag notes (§2 covers callers):
- `compute_pmf_ffi`: docstring claims "Differentiable with custom VJP" (`:566`) but is a bare `ffi_call` (`:585-600`) — the VJP lives in `pmf_from_graph`; docstring default `granularity=100` contradicts the actual `0` (`:523` vs `:543-544`).
- `compute_pmf_multivariate_ffi`: `compute_joint=True` → `ValueError` "not yet implemented" (`:877-882`); silently transposes rewards (`:901-908`).

---

## 5. CELLS EXPECTED TO RAISE — guard conditions (for a regression harness)

A harness distinguishes an intended raise from a regression by matching the exception TYPE + the guard `file:line`. Grouped by layer.

### Python argument/shape guards
| Guard | Exception | file:line |
|---|---|---|
| JAX not importable (pmf_from_graph) | `ImportError` | `:3555-3560` |
| callback mode, no `weight_callback` (pmf_from_graph) | `ValueError` | `:3596-3600` |
| param graph + ffi disabled (pmf_from_graph) | `PTDBackendError` | `:3674-3686` |
| zero non-start vertices | `ValueError` | `_graph_serialize.py:42-43` |
| inconsistent coefficient lengths | `ValueError` | `_graph_serialize.py:104-110` |
| pmf_from_cpp missing file | `FileNotFoundError` (before JAX check) | `:4022-4023` |
| pmf_from_cpp no `build_model` substring | `ValueError` | `:4038-4041` |
| reward_visit rewards shape | `ValueError` | `:3304-3308` |
| moments/pmf_and_moments no param edges | `ValueError` | `:6457-6461 / 6683-6687` |
| joint_index no param edges | `ValueError` | `:7271-7275` |
| joint_prob_graph param_length==0 | `ValueError` | `:8339-8340` |
| joint_prob_graph both limits default | `ValueError` | `:8341-8342` |
| joint_prob_graph reward_rates_callback set | `AttributeError` (dead delegate) | `:8381` |
| joint_stop/sojourn not a joint graph / ≠2 trash / no absorbing | `ValueError` | `:9029-9077 / 9214-9259` |
| daisy: not JSP / shape / t_eval<=0 / bad final_read / no `_joint_prob_source` | `ValueError` | `:9469-9556` |
| update_weights theta 1-D/empty/NaN/inf/both-modes | `ValueError` | `:1925-1935` |
| pdf NaN/negative time, bad granularity | `ValueError`/`TypeError` | `:2296-2303` |
| moments(discrete=True) on continuous | `ValueError` | `:2110-2111` |
| SVGD scalar guards (n_particles/n_iterations/lr/reg/nr_moments) | `ValueError` | `:5501-5513` |
| observed_data not 1-D / NaN / Sparse-without-rewards | `TypeError`/`ValueError` | `:5653-5666` |
| theta_dim un-inferable | `ValueError` | `:5669-5675` |
| joint + regularization / joint + rewards | `NotImplementedError` | `:5756-5764` |
| SVGD config rules R1–R28 | `SvgdConfigError` | `svgd_config.py:689-1100` (per-rule lines in §7) |
| SVGD model not 2-tuple | `ValueError` | `svgd.py:5638-5644` |
| SVGD theta_init not 2-D | `ValueError` | `svgd.py:5429-5433` |
| SVGD exposure ≤0 / ndim>1 / with Sparse | `ValueError`/`NotImplementedError` | `svgd.py:5065-5108` |
| SVGD `jit=True` while config JAX-disabled | `PTDConfigError` | `svgd.py:4905-4910` |
| SVGD access results before fit | `RuntimeError` | `svgd.py:6747` (+ siblings) |

### C++ GraphBuilder / FFI guards
| Guard | Exception | file:line |
|---|---|---|
| `Unknown weight_mode` (e.g. callback reaches FFI) | `std::invalid_argument` → XLA | `graph_builder.cpp:29-36` |
| formula mode without tape / tape without formula mode | `std::invalid_argument` | `graph_builder.cpp:44-60` |
| `len(theta) != param_length` | `std::invalid_argument` "Theta length mismatch" | `graph_builder.cpp:172-179` |
| log mode + non-positive `coeff*theta` | `std::invalid_argument` | `graph_builder.cpp:449-457` |
| formula → non-finite/negative weight | `std::invalid_argument` | `graph_builder.cpp:443-445` |
| **`nr_moments=0` → SEGFAULT (no guard)** | exit 139 | `graph_builder.cpp:480-491` |
| theta/times/rewards rank or batch mismatch | `ffi::Error::InvalidArgument` | `graph_builder_ffi.cpp:139,150,172-175,580,592,604,804,813,826,951,1414-1445,1737-1744` |
| negative index (sojourn) | `ffi::Error::InvalidArgument` | `graph_builder_ffi.cpp:837` |
| daisy: missing `_daisy_chain` / n_epochs / dts length / theta length / ipv length | `ffi::Error::InvalidArgument` | `graph_builder_ffi.cpp:1340,1697,1709-1744` |
| **negative edge weight < -1e-12 (daisy ONLY)** | `ffi::Error::InvalidArgument` | `graph_builder_ffi.cpp:1238,1546-1552,1632-1640,1783-1834,1870-1878` |
| malformed JSON | `ffi::Error::InvalidArgument` | `graph_builder_ffi.cpp:113-117,1682-1695` |

### C-core guards
| Guard | Exception | file:line |
|---|---|---|
| **DPH row sum > 1.0001** (absolute tol; any dph read) | C NULL+`ptd_err` → `std::runtime_error`/`XlaRuntimeError` "outgoing rate <= 1" | `src/c/phasic.c:11172-11208` (`if (rate > 1.0001)` at `11182`); `phasiccpp.h:1341-1352,43-48` |
| param length mismatch (linear/log update_weights) | `ptd_err` "Parameter length mismatch" | `src/c/phasic.c:5565-5572` |
| trace replay theta length | `ptd_err` "Expected N parameters" | `src/c/phasic.c:12351-12355` |
| trace recording on non-param graph | `ptd_err` | `src/c/phasic.c:12689-12690` |
| add_edge coefficients shorter than param_length (longer is allowed) | `ptd_err` "too few coefficients" | `src/c/phasic.c:4836-4848` |
| trace eval NaN (e.g. zero exit rate) | `ptd_err` "numerical catastrophe" | `src/c/phasic.c:12504` |
| C++ IPV: entry ≤0 / sum > 1+1e-9 | `std::invalid_argument` | `phasiccpp.cpp:198-214` |
| Python IPV validation | `TypeError`/`ValueError` | `__init__.py:1141-1199` |

**NOT guarded (regressions here will NOT raise):** negative edge weights outside daisy (`src/c/phasic.c:5745-5755` commented out); NaN/Inf theta at C level (`src/c/phasic.c:5573-5592` commented out — only the Python guards at `__init__.py:1929-1932` catch it); `nr_moments=0` (segfault, not an exception); all-zero-state second vertex (crashes as `SystemError` on the generated-C++ branch `:3801/4128/4117`, or as `XlaRuntimeError` NaN catastrophe on the FFI branch — see Contradiction 4).

---

## 6. `Graph.svgd` FLAG MATRIX (entry #13) — condensed

**Model dispatch (`:5716`)** keys ONLY off graph kind + `epoch_starts` + `rewards.ndim`; the `joint_index` kwarg never selects a builder.
- `_joint_prob_base_graph_indexer` set → JOINT path (`:5716`): `epoch_starts` set → `_daisy_chain_svgd_model` (`:5833`); else → `pmf_from_graph_joint_index` (`:5881`).
- else `rewards.ndim==2` → `pmf_and_moments_from_graph_multivariate(use_ffi=False)` (`:5892`); `rewards` 1D/None → `pmf_and_moments_from_graph` (`:5899/5905`).

**Compute flags:** `observed_data`, `discrete`, `epoch_starts`, `final_read` (daisy only), `daisy_chain_t_eval`/`_granularity`/`_probe_theta`/`_t_eval_tol` (daisy only; auto-probe only when `='auto'`), `rewards`, `validate_rewards`, `fixed`, `tied` (daisy only), `exposure`+`exposure_param_index`, `theta_dim`, `prior`, `regularization`, `nr_moments`, `callback`, `weight_formula`, `positive_params`/`param_transform`, `preconditioner`.
**Execution-only:** `optimizer`/`learning_rate`, `jit`/`parallel`/`n_devices`/`precompile`, `n_particles`/`n_iterations`/`bandwidth`/`seed`/`return_history`/`progress`/`verbose`/`compilation_config`/`quiet_assumptions`.

**RAISES — SVGD config rules (all `SvgdConfigError` unless noted), from `svgd_config.py`:** R1 epoch_starts on non-joint / discrete-joint `:692-708` (also `ValueError` at `__init__.py:4267-4273`); R2 joint_index=True on non-joint `:711`; R3 rewards on joint `:722` (also `NotImplementedError :5760`); R4 regularization on joint `:731` (also `:5756`); R5 Sparse without rewards `:740`; R6 exposure+Sparse `:749` (`NotImplementedError svgd.py:5066`); R7 exposure/index pairing `:760-771`; R8 index out of range `:774-792`; R9 exposure+joint without epoch_starts `:795-809`; R10 exposure length `:812-826`; R11 exposure+2D-rewards warn `:829`; R12 param_transform on joint `:841-851`; R13 fisher+joint warn `:854`; R14 fixed local index under daisy `:873-887` (`ValueError :4336`); R15 positive_params+param_transform `:890-898` (`ValueError svgd.py:5212`); R16 tied without epoch_starts `:901-907`; R17–R20 tied structure `:910-1022`; R21 callback+epoch_starts `:959-967`; R22 weight_formula+callback `:970-977`; R23 fixed out-of-range non-daisy `:1025-1038`; R24 unknown preconditioner `:1041-1053`; R25 optimizer+learning_rate `:1056-1062` (`ValueError svgd.py:5156`); R26 optimizer+regularization `:1065-1072` (`svgd.py:5161`); R27 nr_moments≠2 on joint warn `:1075-1085`; R28 joint_index=False on joint `:1088-1098`.
Plus scalar guards `:5501-5513`, observed_data guards `:5653-5666`, theta_dim `:5669-5675`, callback/weight_formula theta_dim `:4987-5010`, `final_read` (daisy) `:4448-4451`, `daisy_chain_t_eval` string/sign `:9821-9830`, epoch_starts monotonicity `:4240-4249`, negative-rate daisy `graph_builder_ffi.cpp:1238`.

**Silent overrides:** see §3 #8, #9, #11, #12, #14, #15, #23, #24, #26, #28. Additionally: `prior=None` (no epoch_starts) builds a `DataPrior(sd=5)` (`:5693-5713`), and if that raises it silently falls back to standard normal with an `SvgdAssumptionWarning`; `prior=None`+epoch_starts runs `probability_matching` and silently returns None (standard normal) on any exception (`:4938-4941`); `tied` slaves are appended to `fixed` with sentinel 0.0 (`:4437`) which is visible in `svgd.fixed`/`effective_options()` though meaningless. Verify what fired via `res.effective_options()` (ledger built `:5561-5573`).

**Builder overwrite (entry #17):** `_daisy_chain_svgd_model` OVERWRITES `Graph.svgd`'s resolved `theta_dim` (→ `n_epochs*param_length`, `:4318`), `prior`, and `fixed` (`:5833-5858`, announced via `SvgdAssumptionWarning`+ledger).

> Note: `SVGD.__init__(prior=None)` is a plain standard normal (`svgd.py:6086-6087`), whereas `Graph.svgd(prior=None)` builds a `DataPrior(sd=5)` — a documented, deliberate difference.
> Potential crash (not read-tested): `Graph.svgd(theta_init=X, n_particles=None)` on a linear/log graph — `theta_dim` stays None (`:5669` gated on `theta_init is None`) and `SVGD.__init__` computes `20*theta_dim` before deriving it from `theta_init` (`svgd.py:4893-4894` vs `5427-5435`).

---

## 7. CONTRADICTIONS BETWEEN SOURCE-READERS — explicitly UNRESOLVED

1. **[UNRESOLVED] `weight_mode` survival into joint-prob graphs (§3 #3).** Pass 4 verified `_weight_mode='log'` → joint graph `'linear'`, concluding a log model silently becomes linear in all joint-prob inference (`:1552`; no propagation in `8312-8703`, `9161-9167`, `9308-9317`). Pass 3 claims linear/log semantics still reach C++ via `serialize()['weight_mode']` and only `formula` needs explicit propagation (`:9173/9319`, `:614-615`). Not reconcilable for the log case without an end-to-end numeric check. **Do not assume log joint-prob inference is correct until resolved.**

2. **[UNRESOLVED — likely reconcilable] `moments_from_graph` implementation.** Pass 2 describes the body as JIT-compiling C++ via `_generate_cpp_from_graph → _compile_wrapper_library → ctypes.PyDLL` (`:6464,6522,6525`, needs `PHASIC_SOURCE_DIR`). Pass 5 describes the body as "unconditionally `jax.pure_callback` at `:6556`" and says `compute_moments_ffi` is never referenced. Both agree `use_ffi` is a dead/misleading flag and `PHASIC_SOURCE_DIR` is required; the likely truth is a `pure_callback` wrapping a ctypes call, but the two descriptions cite different lines and were not cross-checked. Treat the exact call graph as unverified.

3. **[UNRESOLVED — minor] `pmf_from_graph_parameterized` failure mode.** Pass 1 reports TWO alternative failures at `:535` (AttributeError when JAX never activated in-process; dtype RuntimeError when it was). Pass 4 reports only the dtype RuntimeError "under default config." Both agree the entry point is broken and `:535` is the failing line; the discrepancy is whether construction leaves module globals `jax=None`. Pass 1 is the superset.

4. **[UNRESOLVED — branch-dependent] all-zero-state second vertex, exception type.** Pass 1: on the non-parameterized generated-C++ branch of `pmf_from_graph`, an all-zero-state second vertex aliases onto the start pointer and raises `SystemError: nanobind … exception could not be translated` (frames `:3801,4128,4117`; test `:869-878`). Pass 5: on the FFI/parameterized branch, `find_or_create_vertex([0])` produces `XlaRuntimeError` "Computation produced NaN at vertex 0 - numerical catastrophe." These are different exceptions because they are different branches (generated-C++ vs FFI); a harness must key on the branch. Both trace to the start-vertex aliasing in `_generate_cpp_from_graph` (`:868-877`) / the all-zero absorbing-state convention.

5. **[NOT a contradiction — different fixtures] daisy-chain t_eval-no-op sum.** Pass 3 reports sum `0.6373456790123462`; pass 5 reports `0.9779702315635295`. Different test graphs; both confirm the same qualitative finding (t_eval no-op under `'sojourn'`, bit-identical output).

6. **[Consistent, noted] 2D-rewards default-path failure.** Pass 2 says it RAISES `RuntimeError` "Incorrect output shape"; pass 4 says the output shape silently depends on `weight_mode`. Both stem from the `shape[1]` (`:6949`) vs `shape[0]` (`graph_builder.cpp:666`) mismatch — consistent, complementary framings.

---

*Any claim in the source JSON that lacked a `file:line` citation has been dropped. Numeric values marked "VERIFIED" were executed by the originating reader; FFI-wrapper numeric outputs marked "NOT executed" (`compute_pmf_multivariate_ffi`, `compute_sojourn_times_ffi`, the daisy wrappers called directly) are guard-grounded but their forward numbers are unverified.*

<!-- END source-derived flag matrix -->
