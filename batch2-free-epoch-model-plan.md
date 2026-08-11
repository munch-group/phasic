# Plan — Batch 2: public free daisy-chain epoch model builder

> **Plan-review outcome: SOUND-WITH-GAPS → 6 amendments folded in (v2).**
> 1. **FD-skip: bake NOTHING** (reviewer proved bake-nothing is *bit-identical* to bake-base on the free slots, since SVGD never uses fixed-slot gradients) — eliminates the reuse footgun and the superset constraint. Implement via an additive `bake_fd_skip=False` kwarg on `_daisy_chain_svgd_model` (default `True` preserves `Graph.svgd`).
> 2. **`.fit(fixed=)` uses FLAT `(flat_idx, value)` indexing** (not local/per-epoch). The same-model-LRT `df=1` requires it; documented asymmetry vs `epoch_model(fixed=)` which is local/per-epoch-broadcast (mirroring `Graph.svgd`).
> 3. **`.fit()` skips prior re-masking when the prior is `None`** (`_daisy_chain_svgd_model` returns `None` on `probability_matching` failure, `:4962`).
> 4. **Exposure is fully baked**; `.fit()` does NOT pass `exposure=` to SVGD. Exposure parity is unproven until the 2c exposure test (E1/E2 were no-exposure).
> 5. **Factual fix:** the observation mapping is NOT unique to `Graph.svgd` — it is duplicated in `Graph.probability_matching` (`__init__.py:6378-6391`). FD-skip lives at `:4557/4760` (`fixed_set_local`) + skip loops `:4607/4847` (earlier draft cited `:4634/4874`, the tag comments).
> 6. **2a gate = bit-identity for `seed=None`** (confirmed: the RNG tie-break fires only on degenerate groups, which the epoch fixtures never hit → zero global-RNG draws; `seed=None`→global `np.random`, a supplied `seed`→`RandomState(seed)`).

## Goal

Add a **public** way to build a reusable *free* (untied) daisy-chain (epoch / time-inhomogeneous) model, closing the asymmetry with the plain path (which has public `Graph.pmf_and_moments_from_graph`). Today the only way to build an epoch model is `Graph.svgd(epoch_starts=…)`, which builds it internally via the private `_daisy_chain_svgd_model` and fits immediately — you cannot get the model object back.

**Value:**
1. **Reuse** — build the (expensive) JSP graph + model once, run many fits (seeds / hyper-params) without rebuilding.
2. **`log_likelihood(theta=…)`** on a fitted free epoch model (verified working).
3. **Same-model LRT fast path** — two fits from the SAME free-model object (`full` free, `nested` = free + extra `fixed`) share `.model`, so `likelihood_ratio_test(full, nested)` takes the structural fast path (df from fixed-masks). This complements **Batch 1** (which handles *tied*-vs-free via the different-callable path); Batch 2 handles *free*-vs-*fixed* nesting on epoch models.

Out of scope: tied fits (still `Graph.svgd(tied=…)`); the rejected A-move (SVGD-level tying).

## De-risk findings (experiments E1/E2 run against the installed build; PREMISE HOLDS-WITH-FRICTION)

- **E1 (reproduce):** free model built out-of-band (replicating the `Graph.svgd` epoch preamble) + a **direct** `SVGD(...)` reproduces `Graph.svgd(epoch_starts=…, tied=None)` **to machine precision**: `theta_dim=4`, `dof=2`, MAP max-abs-diff **0.0**, `log_likelihood()` diff **0.0**. `SVGD.log_likelihood(theta=MAP)` works on the free fit.
- **E2 (same-model LRT):** two direct fits from the SAME `model` object → `full.model is nested.model` True (fast path), `dof` 2 vs 1 → **df=1**, coherent `LRTResult`.
- **Friction the builder must own (all plumbing, no new numerics):**
  1. **Observation mapping** (raw joint-outcome tuples → vertex indices): `joint_prob_table().groupby(...)` + `np.random.choice` tie-break for degenerate groups. Present in `Graph.svgd` (`__init__.py:5757-5772`) AND duplicated in `Graph.probability_matching` (`:6378-6391`) (amendment 5). `_daisy_chain_svgd_model` expects already-mapped vertex indices (`:4318-4331`) and errors otherwise.
  2. `SVGD(observed_data=…)` must receive the **1-D mapped indices**, not the raw tuples (svgd.py:5970 shape check). The daisy model ignores its `observed_arg` (positions baked at build, `:4621/:4861`), but SVGD uses `observed_data` as a shape/NaN-mask carrier (svgd.py:5964-5991) — so the builder must **surface** the mapped indices.
  3. **Per-fit prior re-masking** — a shared prior list is rejected by SVGD when a fit fixes an extra slot (`prior[i] provided but theta[i] fixed`, svgd.py:5413). Each fit needs `None` at its fixed indices.
- **RNG determinism concern (surfaced by the mapping):** the degenerate-group tie-break consumes global numpy RNG, so two *separate* builds could map differently. For the primary use (build once → fit many) this is irrelevant (one mapping, shared). But reproducibility / cross-build comparison want determinism → the builder should accept a `seed` (or use a deterministic tie-break) for the mapping.

## API design

Return a small object, not a bare tuple (the 4-tuple `_daisy_chain_svgd_model` returns omits the mapped `observed_data` and forces the caller to re-mask priors — the friction above):

```python
class FreeEpochModel:
    model            # the JAX-differentiable free daisy-chain callable (carries _tying_info={}, tags)
    theta_dim        # n_epochs * param_length
    observed_data    # 1-D mapped vertex indices (feed straight to SVGD)
    prior            # broadcast per-slot prior list (fixed slots already None)
    fixed            # broadcast (flat_idx, value) list
    n_epochs, param_length, t_eval, epoch_starts
    def fit(self, *, fixed=None, prior=None, **svgd_kwargs) -> "SVGD":
        # fixed uses FLAT (flat_idx, value) indexing (amendment 2); merged with
        # the builder's base fixed. prior re-masked at the union of fixed indices
        # UNLESS it is None (amendment 3). exposure is NOT passed (fully baked,
        # amendment 4). Constructs SVGD(model=self.model, observed_data=..., ...),
        # optimize(), returns the fitted SVGD. Because the model bakes NO FD-skip
        # (amendment 1), any fixed set is correct — no superset constraint.
```

Public entry point (name TBD — recommend **`Graph.epoch_model`** for discoverability; alt `Graph.pmf_from_graph_joint_index_epochs` to match the plain-path naming convention):

```python
def epoch_model(self, observed_data, epoch_starts, *,
                prior=None, fixed=None,
                exposure=None, exposure_param_index=None,
                daisy_chain_t_eval=None, daisy_chain_probe_theta=None,
                daisy_chain_t_eval_tol=1e-3, daisy_chain_granularity=0,
                final_read='sojourn', seed=None, verbose=False) -> FreeEpochModel
```
No `tied` (free-only, by design). Signature mirrors the `Graph.svgd` daisy kwargs so behaviour is identical.

## Approach (batched, test-gated)

> Rebuild reality: pure-Python edits still need `pixi run install-dev` (copy install). `PHASIC_SOURCE_DIR=/Users/kmt/phasic`. Never `git add -A` / `git stash` / touch `.ipynb`. Base the branch off master (`e42f46fa`).

### Batch 2a — extract the observation-mapping into a shared private helper (DRY on the risky bit)
- Extract the inline mapping into `Graph._map_joint_observations_to_indices(observed_data, *, seed=None) -> indices_1d`. `seed=None` → global `np.random` (bit-identical to today, incl. the degenerate-group tie-break); a supplied `seed` → local `np.random.RandomState(seed)` for determinism (amendment 6).
- Replace the inline block in `Graph.svgd` (`:5757-5772`) with a call to it (≈1-line change). There are ALREADY two copies (`Graph.svgd:5760` and `Graph.probability_matching:6378`, amendment 5); optionally retire the second copy too (extra DRY vs a second edit to existing code — decide in review). At minimum, do not add a third.
- **Gate (amendment 6):** the `seed=None` extraction must be **bit-identical** — `test_lrt_at.py` (6) + the joint-index/daisy tests (`test_gate_daisy_chain_joint_probs.py`, `test_joint_index_callback.py`, `test_optimized_joint_index.py`) stay green. (The tie-break fires only on degenerate observation groups, which the epoch fixtures never hit → zero global-RNG draws, so `seed=None` consumes the stream identically.)

### Batch 2b — `_daisy_chain_svgd_model` gains `bake_fd_skip=False` + the public builder + `FreeEpochModel` + `.fit()`
- **`_daisy_chain_svgd_model`**: add additive kwarg `bake_fd_skip: bool = True` (default preserves `Graph.svgd`). When `False`, still compute `broadcast_fixed`/masked-prior for the RETURN, but do NOT add `fixed_indices` to `fixed_set_local` (`:4557/4760`) — so the FD backward computes real gradients at all slots (bit-identical to today on free slots; only the discarded fixed-slot grads change). `epoch_model` calls with `bake_fd_skip=False` (amendment 1).
- New `Graph.epoch_model(...)`: (1) validate (reuse `svgd_config` epoch/exposure rules), (2) map observations via the 2a helper (`seed`), (3) `resolved_t_eval = self._resolve_daisy_chain_t_eval(...)`, (4) build `_daisy_exposure` (scalar→per-obs, `:5841-5853`), (5) `_daisy_chain_svgd_model(user_tied=None, bake_fd_skip=False, …)`, (6) wrap in `FreeEpochModel` (surfacing mapped `observed_data`).
- `FreeEpochModel.fit(fixed=…, prior=…, **svgd_kwargs)`: merge base + extra `fixed` in **FLAT** index space (amendment 2); re-mask the prior list at the union of fixed indices **only if the prior is not None** (amendment 3); do NOT pass `exposure=` (baked, amendment 4); `SVGD(model=self.model, observed_data=self.observed_data, theta_dim=self.theta_dim, prior=masked_prior, fixed=merged_fixed, **svgd_kwargs).optimize()`.
- `pixi run install-dev`.

### Batch 2c — tests (`tests/pytest/inference/test_epoch_model.py`)
- **reproduce:** `epoch_model(...)` + `.fit()` == `Graph.svgd(epoch_starts=…, tied=None)` on the same data/seed — MAP + `log_likelihood()` to machine precision (the E1 gate).
- **reuse / same-model LRT:** one `epoch_model`, two `.fit()`s (free vs +fixed) → `full.model is nested.model` True; `likelihood_ratio_test(full, nested)` df=1 via the fast path (the E2 gate).
- **log_likelihood(theta=…)** on a fitted free epoch model.
- **exposure:** `epoch_model(exposure=…, exposure_param_index=…)` matches `Graph.svgd(exposure=…)`.
- **prior re-masking:** `.fit(fixed=extra)` succeeds where passing the raw shared prior would raise.
- **determinism:** two builds with the same `seed` produce identical mapped `observed_data` (guards the RNG concern).

### Batch 2d — adversarial review of the diff
Refute: (a) the 2a extraction changed any existing epoch fit (bit-identity vs `Graph.svgd`); (b) `.fit()`'s prior re-masking / fixed-merge is correct for base∩extra overlaps; (c) the mapped `observed_data` shape/NaN handling matches what SVGD expects; (d) exposure path parity; (e) `epoch_model` + `.fit()` == `Graph.svgd` across ≥2 fixtures; (f) determinism under seed.

## Verification (end-to-end)
- E1/E2 gates as regression tests; `epoch_model().fit()` == `Graph.svgd(tied=None)` to machine precision on ≥2 models; same-model LRT df correct; existing daisy-chain/epoch tests green after the 2a extraction.

## Risks / notes
- **Modifying `Graph.svgd` (2a).** Minimal (one call-site swap) and justified by DRY on the RNG-sensitive mapping; gated by existing epoch tests + adversarial bit-identity check. Tension with the standing "don't modify existing" preference — the conservative alternative (duplicate the mapping in the builder, zero `Graph.svgd` change) is viable but risks the two copies diverging exactly on the subtle RNG tie-break. **Recommend the extraction; flag for approval.**
- **RNG tie-break determinism.** Default behaviour must be decided: keep global-RNG (current, non-deterministic under degeneracy) for exact back-compat, or make `seed`-deterministic the default (cleaner, but changes `Graph.svgd`'s mapping under degenerate groups — re-baseline needed). Lean: add `seed`, default `None` = current behaviour; deterministic only when a seed is given.
- **Naming** (`epoch_model` vs `pmf_from_graph_joint_index_epochs`) — user's call.
- **`.fit()` scope creep.** The convenience `.fit()` is what turns the multi-step friction into one call and is what makes the same-model LRT ergonomic; without it the builder is a leaky 5-tuple. Keep it, but it is the main new surface to review.
- **FD-skip invariant (RESOLVED by amendment 1 — bake nothing).** `_daisy_chain_svgd_model` normally bakes build-time `user_fixed` into the FD backward as a 0-gradient skip (`fixed_set_local` `:4557/4760`, skip loops `:4607/4847`), so a reused model baked with `fixed=[(1,mu)]` would silently mis-fit any `.fit(fixed ⊉ {mu})` (mu's gradient forced 0). The plan review empirically confirmed the footgun AND that **bake-nothing is bit-identical on the free slots** (SVGD reduces to the learnable subspace and never reads a fixed slot's gradient, `svgd.py:3971/3996/4114`). So `epoch_model` builds with `bake_fd_skip=False`: no skip, any `.fit(fixed=…)` set is correct, no superset constraint. `Graph.svgd` keeps the default `bake_fd_skip=True` (unchanged).

## Rejected alternatives
- **Bare 4-tuple `(model, theta_dim, prior, fixed)`** (mirror the private helper) — insufficient: omits the mapped `observed_data`, forces the caller to redo the mapping and re-mask priors (the exact E1/E2 friction).
- **Full duplication of the preamble in the builder, zero `Graph.svgd` change** — additive and respects "don't modify existing", but duplicates the RNG-sensitive mapping (divergence risk). Fallback if the extraction is deemed too invasive.
- **Refactor `Graph.svgd`'s whole epoch branch to delegate to `epoch_model`** — maximal DRY, but the tied path complicates it (builder is free-only) and it is a larger edit to load-bearing code than warranted. The 2a surgical extraction captures most of the DRY benefit at a fraction of the risk.
