# Deferred: likelihood-ratio testing / model reuse on epoch (daisy-chain) joint-prob SVGD models

**Status:** deferred — analysed, not fixed. This document is self-contained; it
records the user scenario, the exact blockers, verified code references, a
candidate near-term workaround, and fix options with acceptance criteria.

**One-line:** You cannot currently run a nested likelihood-ratio test (LRT) —
or otherwise reuse one fitted model's `SVGD.log_likelihood()` — across an
all-tied vs coalescent-free **epoch (daisy-chain) joint-probability** pair,
because (1) the direct `SVGD(model=…)` path rejects the epoch/tying options by
design, (2) epochs/tying/exposure are baked *into the model callable*, and
(3) the LRT requires both fits to share the *same* model callable while
expressing nesting through `fixed`, which cannot represent a tying restriction.

---

## 1. The scenario

Time-inhomogeneous coalescent joint-probability inference (see the
`svgd-joint-prob.ipynb` tutorial). The user fits two nested models over the same
epochs and data:

- **nested** — coalescent rate *tied* across epochs (`svgd_all_tied`);
- **full** — coalescent rate *free* per epoch (`svgd_all_coal_free`).

and wants to compare them with a Wilks likelihood-ratio test,
`2·(LL_full − LL_nested) ~ χ²_df`, via `phasic.model_selection.likelihood_ratio_test`.
That requires `SVGD.log_likelihood()` on both fits.

The user constructed the second fit by **passing the already-built model to the
`SVGD` constructor directly** (rather than re-calling `Graph.svgd`) and hit:

```python
svgd_all_coal_free = SVGD(
    model=svgd_all_tied.model,
    observed_data=svgd_all_tied.observed_data,   # reuse processed (mapped) obs
    ...,
    epoch_starts=epoch_starts,
)
# TypeError: SVGD.__init__() got an unexpected keyword argument 'epoch_starts'
```

**Why reuse the model at all (not re-call `Graph.svgd`)?** Because
`likelihood_ratio_test` *requires it* — see §2.

---

## 2. What the LRT API requires (and why model reuse is mandatory)

`phasic.model_selection.likelihood_ratio_test(full, nested, …)`
(`src/phasic/model_selection.py`) enforces, in strict mode:

- **Same model callable** — `if full.model is not nested.model: _violate(...)`
  (identity check, ~`model_selection.py:413`). Two separate `Graph.svgd(...)`
  calls build two *different* callables, so they are rejected.
- Same `theta_dim`, same `n_observations`.
- **Nesting expressed via `fixed`** — `_check_nested_fixed` requires the nested
  fit to fix a *superset* of the full fit's fixed parameters (same values).
- df = `k_full − k_nested` (from `degrees_of_freedom`), and it rejects
  `LL_nested > LL_full` (LL inversion).

The **canonical pattern** the docstring prescribes (`model_selection.py:346`):

```python
model  = Graph.pmf_and_moments_from_graph(graph)          # ONE model callable
full   = SVGD(model, data, theta_dim=k).optimize()
nested = SVGD(model, data, theta_dim=k, fixed=[(0, 1.0)]).optimize()
res    = likelihood_ratio_test(full, nested)
```

So the intended design is: **build one model, wrap it in two `SVGD` instances
that differ only by `fixed`.** The user's "reuse `svgd_all_tied.model`" is a
direct (and correct) attempt to satisfy the `full.model is nested.model`
identity requirement. It works for ordinary models. It does **not** work for
epoch / joint-prob models, for the reasons below.

---

## 3. The blockers

### Problem 1 — the direct `SVGD(model=…)` path rejects `epoch_starts` (by design)

`SVGD.__init__` (`src/phasic/svgd.py:4763`) accepts the inference options
(`prior, n_particles, learning_rate, fixed, exposure, exposure_param_index,
optimizer, preconditioner, theta_dim, …`) but **not** `epoch_starts`, `tied`,
`joint_index`, `discrete`, `callback`, or any `daisy_chain_*`. The class
docstring states this explicitly (`svgd.py:4387–4404`):

> Parameters unique to `Graph.svgd` — `discrete`, `joint_index`, `tied`,
> `epoch_starts`, `daisy_chain_t_eval`, … — are **pre-model-construction
> concerns and have no meaning when SVGD is handed an opaque model callable.**

Rationale is structural: `epoch_starts` *builds* the daisy-chain model —
`Graph.svgd` consumes it to call `self._daisy_chain_svgd_model(...)`
(`src/phasic/__init__.py:6876–6917`), which needs the `Graph`. `SVGD` only
receives the finished callable, so it cannot act on `epoch_starts`. The model
being reused *already encodes* the epochs (its `theta_dim` is already the flat
`n_epochs × param_length`).

→ The `TypeError` is by design. But note it is a *bare* `TypeError`, with no
hint toward the supported route (Problem 4).

### Problem 2 — epochs / tying / exposure are baked *into the model callable*

`_daisy_chain_svgd_model` (`src/phasic/__init__.py:5242`) bakes the full
configuration into the returned callable:

- **tying:** `model._tying_info = tying_info` (`__init__.py:5958`), and
  `theta_arr = _apply_tying(theta_arr)` is applied on **every forward call**
  (`__init__.py:5678`, `:5918`);
- **exposure:** `model._handles_exposure_internally = True`
  (`__init__.py:5690`, `:5929`);
- **epochs:** the per-epoch handoff structure and flat theta layout.

Consequences:

- `svgd_all_tied.model` is the *tied* model. Reusing it — even if you drop
  `epoch_starts` and pass `theta_dim` — keeps collapsing the coalescent slave
  columns onto their master on every call, so you get the **tied** likelihood
  back, not a free one. It cannot serve as the "full/free" model.
- The tied model and a free model are therefore **different callables**, so
  `full.model is nested.model` can never hold for a tied-vs-free pair built the
  normal way.
- There is **no public standalone builder** for a daisy-chain epoch model to
  hand to two `SVGD` constructors (`_daisy_chain_svgd_model` is private and
  only reachable through `Graph.svgd`). The canonical LRT pattern's first line
  (`model = Graph.pmf_and_moments_from_graph(graph)`) has no daisy-chain
  equivalent.

### Problem 3 — tying-nesting cannot be expressed as `fixed`-nesting

The LRT expresses nesting through `fixed` (nested fixes a superset of full's
fixed params). The user's nesting is a **tying** restriction (tie the
coalescent rate across epochs; the full model frees it). Because tying is baked
into the model rather than applied as an SVGD-level `fixed`/transform, "same
model, nested = extra `fixed`" simply cannot encode "tied vs free." The two
restrictions live at different layers.

### Problem 4 — misleading docstring surface

The `SVGD` class docstring (`svgd.py:4380`) is long and includes "Combination
with `tied`" / `epoch_starts=[…]` example blocks (shared with `Graph.svgd`'s
docs) that read as if the direct path accepts them, even though the "When to
use" header says the opposite. This is almost certainly what set the
expectation that "`SVGD` shares epoch options with `Graph.svgd`." The
`__init__.py:6380` note ("Not available on the direct `SVGD(model=…)` path") is
correct but easy to miss.

---

## 4. Candidate near-term workaround (UNVERIFIED — validate before relying on it)

A rigorous nested LRT only needs **one** likelihood function evaluated at two
points: the free MLE/MAP and the best constrained (tied) point. A free model
can represent the tied one by setting the coalescent params equal across epochs.
So, using the **free** fit's model only:

```python
free = joint_prob_graph_cont.svgd(..., <coalescent free>).optimize()   # full model
ll_full   = free.log_likelihood(refine=True)                            # at free MAP

# theta_nested: the free theta_dim vector with the coalescent slots tied
# (set equal across epochs) at the constrained MAP — obtain by fitting the
# tied model and embedding its MAP into the free (flat n_epochs x param_length)
# layout, OR by re-optimising `free` under the tying constraint.
ll_nested = free.log_likelihood(theta=theta_nested)                     # same callable
stat = max(0.0, 2.0 * (ll_full - ll_nested)); df = k_full - k_nested
# p = scipy.stats.chi2.sf(stat, df)
```

This trivially satisfies "same model" (there is only one), and matches how
`log_likelihood(theta=…)` is meant to be used (constrained space, no
`param_transform` re-applied — `svgd.py:6878`). Open questions to verify:

1. Does `free.log_likelihood(theta=…)` accept a flat `n_epochs × param_length`
   vector and evaluate the daisy-chain likelihood correctly? (Should, since
   that is `free.theta_dim`, but confirm.)
2. How to construct `theta_nested` from a tied fit's MAP — i.e. the exact
   free-layout embedding of the tied parameters. Needs a documented helper.
3. `likelihood_ratio_test(full, nested)` will **not** accept this (it wants two
   `SVGD` objects with `full.model is nested.model` and `fixed`-based nesting),
   so the statistic must be assembled by hand — motivating fix option C.

Cross-reference: re-running `Graph.svgd` is cheap again after the O(n) sojourn
fix (`sojourn-fix.md`), so the old motivation to avoid rebuilding via model
reuse is weaker — but the `full.model is nested.model` identity requirement
remains the hard blocker, independent of cost.

---

## 5. Fix options (for whoever picks this up)

**A. Make tying an SVGD-level restriction on a single reusable model (most
correct).** Add a public builder that returns a *free* daisy-chain epoch model
(e.g. `Graph.pmf_..._from_graph_joint_index_epochs(...)` or a
`return_model=True` mode on `Graph.svgd`), and move tying (and ideally
exposure) *out* of the baked model into an SVGD-level theta transform, so one
model callable serves both fits and `tied`/`fixed` select the restriction at
fit time. Then the canonical pattern works verbatim and
`likelihood_ratio_test(full, nested)` accepts a tied-vs-free pair. Largest
change; unlocks proper nested LRTs generally.

**B. `SVGD.from_fitted(other, *, fixed=…, tied=…)` / `svgd.refit(...)`.** A
classmethod/method that clones a fitted `SVGD` reusing the *same* `model`
object, `observed_data`, and `theta_dim`, changing only the restriction — so
`full.model is nested.model` holds by construction. Cleanly serves
`fixed`-based nesting; tied-vs-free still needs A (since tying is in the model).

**C. First-class one-model-two-theta LRT.** Add
`likelihood_ratio_test_at(svgd, theta_full, theta_nested, df=…)` (or let the
existing function accept a single fitted `SVGD` plus two thetas) so the §4
workaround is supported and validated (LL inversion check, df, p-value) instead
of hand-rolled.

**D. Minimal ergonomics (do regardless).** (i) Make `SVGD.__init__` raise a
*helpful* error for pre-construction kwargs — e.g. "`epoch_starts` is a
`Graph.svgd` option; the model already encodes epochs — build with
`Graph.svgd(...)` or pass `theta_dim=n_epochs*param_length`; see …" — instead of
a bare `TypeError`. (ii) Scope the `tied`/`epoch_starts` blocks in the `SVGD`
class docstring so they clearly point back to `Graph.svgd` and don't read as
direct-path parameters (Problem 4).

---

## 6. Key code references (verified 2026-07-06)

| What | Location |
|---|---|
| `SVGD.__init__` — no `epoch_starts`/`tied` | `src/phasic/svgd.py:4763` |
| `SVGD` docstring "When to use … direct" | `src/phasic/svgd.py:4387–4404` |
| `SVGD.log_likelihood` | `src/phasic/svgd.py:6852` |
| `likelihood_ratio_test` + `full.model is not nested.model` guard | `src/phasic/model_selection.py` (~`:330`, guard `~:413`) |
| Canonical LRT pattern (one model, two `fixed`) | `src/phasic/model_selection.py:346` |
| Daisy-chain model builder (private) | `src/phasic/__init__.py:5242` (`_daisy_chain_svgd_model`) |
| Tying baked into model (`_tying_info`, `_apply_tying`) | `src/phasic/__init__.py:5958`, `:5678`, `:5918` |
| Exposure baked into model | `src/phasic/__init__.py:5690`, `:5929` |
| `epoch_starts` consumed / model built in `Graph.svgd` | `src/phasic/__init__.py:6876–6917` |
| "Not available on the direct `SVGD` path" note | `src/phasic/__init__.py:6380` |

---

## 7. Acceptance criteria for the eventual fix

- A tied-vs-free (or any nested) **epoch joint-prob** pair can be compared with
  a single supported call, with correct df, no LL inversion, and both fits
  provably on the same likelihood function — either via
  `likelihood_ratio_test(full, nested)` (fix A/B) or a documented
  one-model-two-theta path (fix C).
- Reusing a fitted epoch model for `SVGD.log_likelihood()` has a documented,
  working route (no `epoch_starts` `TypeError`; the model's epoch/tying/exposure
  semantics are unambiguous).
- `SVGD.__init__` fails pre-construction kwargs with an actionable message, and
  the `SVGD` docstring matches its signature.
- A regression test covering the epoch-model LRT path (analogous to the existing
  `model_selection` tests).
