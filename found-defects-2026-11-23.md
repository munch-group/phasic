# Defects found during the bug-fix reviews (2026-11-23) — NOT yet approved for fixing

Bugs 1-5 are being fixed under the agreed recipe (plans in the worktrees
`/Users/kmt/phasic-bug{1..5}`). The items below were found by reviewers
along the way, verified to the degree stated, and are listed here for a
decision. None is being changed.

| # | defect | evidence | status |
|---|---|---|---|
| A | **MCMC silently freezes a chain whose initial log-probability is non-finite** (`log_u < nan` is always False, `mcmc.py:563-565, 686-688`); no warning. 17 of 32 chains frozen in one reviewer's runs (BetaPrior case). | `phasic-bug3/experiments/bug3_single_betaprior/review1/` | measured by reviewer |
| B | **Built-in priors hard-code the Jacobian and inverse of ONE transform** (`svgd.py:897, 1095, 1295, 1435, 1671`; samplers `929, 1114, 1304, 1472, 1694`), so any built-in prior combined with a custom `param_transform` has a wrong log-prior and wrong initial points (GaussPrior + `exp`: Jacobian term -0.61 instead of +0.49; initial mean 28 for a prior mean 2.75). | same directory | measured by reviewer |
| C | **`Graph.mcmc` is not reproducible for a fixed seed** (same initial points, different chains on each repeat within one process). | `phasic-bug3/.../review2/r5_graph_mcmc.log` | measured by reviewer |
| D | **`MCMC(parallel='processes')` fails for graph-built models** (`TypeError: cannot pickle ... GraphBuilder`); its only test is commented out. | `phasic-bug2/.../review2/t10.log` | measured by reviewer |
| E | **`SVGD.map_estimate_with_optimization` raises `TypeError`** (passes a Python list to the transform / to JAX), with or without a transform. | `phasic-bug2/.../review2/t2.log` | measured by reviewer |
| F | **Fits with 1-D rewards are not bit-reproducible run to run** (differences ~1e-8 in particles, same seed, fresh model). | `phasic-bug1/.../impl_review/s3b_rewards.py` | measured by reviewer |
| G | **The preconditioner's moment-matching grid is in inverse-softplus units** (`svgd.py:3877-3884`): on a (0,1) slot it cannot reach theta below 0.119. Affects scaling only. | read from source by reviewer | not measured |
| H | **RSS grows ~8-10 MB per fit** in a loop of same-model refits on master and on the bug-1 build alike (not the cache). | `phasic-bug1/.../impl_review/s4b*` | measured, noisy |
| I | `docs/pages/tutorial/model-selection.ipynb` cell 13 uses `GaussPrior(ci=[0.1, 7])` without `lower=0.0` (already raises on master) and fixes migration at 0 (evaluated at 0.693 today; a true zero after bug 2). Notebook — for you to edit. | bug-2 review 3 | read by reviewer |
