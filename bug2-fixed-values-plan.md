# Bug 2: fixed parameter values are not delivered to the model as given — fix plan

**Status: v1, 2026-11-23. Branch `fix/fixed-value-under-positive`, worktree
`/Users/kmt/phasic-bug2` (own pixi env, built from master `ca09e2e1`). No
source changed yet. Backward compatibility is not required (user
decision). Bug 1 (shared compiled-gradient cache) is being fixed on
another branch; experiments here use a fresh model object per fit so that
bug cannot contaminate them (it did, once: b3.log).**

Evidence: `experiments/bug2_fixed_values/` (`b1_what_the_model_sees.py`,
`b1.log`, `b2`, `b3`), unmodified code.

## 1. The defect (read from source + measured)

`SVGD.__init__` stores a fixed value in the UNCONSTRAINED space and lets
the parameter transform map it back (`src/phasic/svgd.py:6149-6196`;
binary-mask form `6197-6225`; the same code in `src/phasic/mcmc.py:321-356`):

```python
if positive_params and value > 0:
    phi_value = float(jnp.log(jnp.exp(value) - 1))     # inverse softplus
else:
    phi_value = value                                   # stored raw
```

and the model later receives `param_transform(phi)`. The round trip is
not the identity. Measured, value the MODEL receives for `fixed=[(1, v)]`
(b1.log; `theta_mean` reports the same wrong number):

| given | setting | model receives |
|---|---|---|
| 0 | positive_params (default) | **0.693147** (= softplus(0)) |
| 1e-12, 1e-10 | positive_params | **1e-9** (the transform's floor) |
| 1e-9 / 1e-8 | positive_params | 1.00000008e-9 / 0.99999999e-8 (`exp(v) - 1` loses digits) |
| 0.3 | positive_params | 0.30000000000000004 |
| 800 | positive_params | **inf** (`exp(800)` overflows), no error |
| -0.5 | positive_params | **0.474** (= softplus(-0.5)), no error |
| 2.0 | custom `param_transform=exp` | **7.389** (= exp(2)) |
| 0 | custom `param_transform=exp` | **1.0** |
| 0.25 | `[BetaPrior, None]`, slot 1 fixed | 0.24999999999999994 |
| 0, -0.5, 2.0 | positive_params=False, no transform | exact |
| binary mask `[0, 1]` | positive_params | 1.0 (exact here) |

Consequences: a nested model with a rate fixed at 0 is evaluated at 0.693;
unscaled rates below 1e-9 cannot be fixed; with a custom transform every
fixed value is wrong.

Related: tied epoch parameters lock each slave slot through this same
mechanism with the value 0.0 as a placeholder (`__init__.py:4723-4735`),
which the model overwrites with the master's value; the placeholder
therefore reaches the model's input as 0.693 today.

## 2. Fix

**Principle: a fixed parameter never passes through a transform. The
model, and every reported result, gets exactly the float the user gave.**

1. In `SVGD.__init__` and `MCMC.__init__`, keep the fixed values in
   PARAMETER space (`self.fixed_theta`, float64 array; exact).
2. After the transform has been chosen (default softplus, per-dimension
   composite, or custom) and the priors have been wired to it, replace the
   model-facing transform by a wrapper that overwrites fixed slots:

   ```python
   base = self.param_transform                  # may be None
   def with_fixed(phi):                          # phi: (..., theta_dim)
       theta = base(phi) if base is not None else phi
       return jnp.where(fixed_mask_bool, fixed_theta, theta)
   ```

   - `self.param_transform` (full-vector, model-facing; used by the
     log-probability, result export, MAP, plots) becomes the wrapper when
     anything is fixed and a transform exists. With no transform nothing
     changes (values are already exact).
   - The unwrapped transform stays available as `self._free_transform`
     and is what priors (`prior._transform`, `svgd.py:6000-6023`,
     `mcmc.py:281-287`) and the preconditioner's learnable-subset path
     (`svgd.py:7260-7288`) use — those apply it to single slots or to a
     learnable-only vector, where the wrapper's mask does not fit. The
     wrapper keeps the `_per_dim_fns` attribute of a composite transform.
   - The wrapper checks `phi.shape[-1] == theta_dim` (a wrong-length input
     is a programming error and raises).
3. The unconstrained particles' fixed columns become a constant
   placeholder (0.0). Nothing reads them as values any more:
   `fixed_values` (unconstrained) stays as the placeholder vector used by
   `svgd_step` / the MCMC proposal to pin those columns.
4. Validation at construction (no silent repair):
   - non-finite fixed value → `ValueError`;
   - negative fixed value with `positive_params=True` → `ValueError`
     (a positive parameter cannot be fixed below zero; 0 is allowed — a
     rate switched off — and is what the tied-slave placeholder uses);
   - the binary-mask form keeps meaning "fixed at exactly 1.0".
5. `fixed` docstrings (SVGD, MCMC, `Graph.svgd`, `Graph.mcmc`,
   `FreeEpochModel.fit`): values are in parameter space and reach the
   model unchanged.

Not changed: how `Graph.svgd` converts per-epoch `fixed` to flat indices;
finite-difference backwards that probe ±1e-7 around a fixed slot at the
few sites that do not skip fixed slots (`pmf_from_graph`, `pmf_from_cpp`,
`moments_from_graph`, reward-visit) — with a value fixed at exactly 0 the
minus probe is negative there (linear weights: the discarded gradient
component is computed from a negative rate; formula/log weights: an
error). That belongs to the gradient-step plan and is listed as a known
limit.

## 3. Tests (new file `tests/pytest/test_fixed_values_exact.py`)

A spy model records the theta it receives (fresh model object per fit).

1. SVGD, positive_params: for v in {0, 1e-12, 1e-10, 1e-9, 1e-8, 0.3, 2,
   50, 800, 1e6}: every value the model receives at the fixed slot `== v`
   bitwise; `theta_mean`, constrained particles, history, `hpd` bounds at
   that slot `== v`; `theta_std == 0`. (Fails on master for all but 2, 50.)
2. SVGD with a custom transform (`exp`) and with `[BetaPrior, None]`:
   exact.
3. SVGD, positive_params=False without transform: exact (passes on
   master; guard).
4. Binary mask: exactly 1.0.
5. MCMC: the same table as 1 and 2 (model input and reported results).
6. Validation: negative under positive_params raises; NaN/inf raise
   (SVGD and MCMC).
7. Free parameters are unaffected: a fit with one slot fixed at an
   ordinary value (2.0) gives bit-identical free-slot particles before
   and after the change (reference numbers stored from the unmodified
   build for one small seeded fit — the only place the old phi value could
   matter is the placeholder, which must not influence free slots).
8. Epoch fits: tied parameters still fit (slave equals master in every
   exported surface, `test_svgd_tied_export.py` unchanged), and a
   per-epoch value fixed at 0 reaches the epoch model as 0.
9. Preconditioner on (`preconditioner='jacobian'`) with a fixed slot: runs
   and the model receives the exact fixed value in the preconditioner's
   own evaluations.
10. MAP / `log_likelihood(refine=True)` keep fixed slots at the exact
    value.

## 4. Gate

Same procedure as bug 1 (`.git/claude-scratch/gate2.sh`,
`gate_compare.py --new-file tests/pytest/test_fixed_values_exact.py`,
`gate_rerun.sh`; per-test comparison against the master baseline; reruns
in isolation of any regression on both builds). Order: edit →
`pixi run install-dev` in this worktree → new tests → gate. Tests that
assert the OLD stored unconstrained value of a fixed slot (e.g. comments
mentioning `inv_softplus(1.0) = 0.541`) are expected to need updating;
each is listed and justified, none is edited just to pass.

## 5. Risks

- Code that reads the unconstrained particles' fixed columns as
  meaningful numbers (plots in unconstrained space, user scripts) now sees
  the placeholder.
- A fixed value of exactly 0 can make a model genuinely degenerate (a
  vertex with no exit); that now surfaces as the model's own error instead
  of silently running at 0.693.
- Rebase: bug 1's branch edits `svgd.py` too; this branch is rebased on
  master after bug 1 merges, and its gate is run after the rebase.
