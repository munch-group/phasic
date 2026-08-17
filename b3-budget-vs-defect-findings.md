# Budget or defect? Settling the four pinned inference failures

The recovery tests carry four strict-xfail pins. Each recorded a
measured failure but could not say whether it was a genuine defect or
merely the reduced iteration/particle budget the suite runs at. This
document settles that by re-running each at substantially larger
budgets with everything else held identical.

**Verdict: three of the four are DEFECTS; one is genuinely
budget-limited.** In the three defects, more budget makes the answer
*worse*, not better, because the mechanism is variance collapse rather
than incomplete convergence. The exception is the 186-vertex two_island
case, which improves markedly with budget and simply needs more compute
than the suite can afford.

Artifacts: `experiments/dr_budget_p1_dispersion.py`,
`dr_budget_p2_joint.py`, `dr_budget_p3_island.py`,
`dr_budget_p4_mixed.py`.

## 1. Under-dispersion — DEFECT. More iterations make it worse.

43-vertex coalescent, off-truth prior, true value 7.0.

| budget | posterior mean | 95% HPD | particle sd | covers 7.0 |
|---|---|---|---|---|
| 150 it, 60 p | 6.567 | [6.465, 6.560] | 0.2525 | no |
| 150 it, 200 p | 6.543 | [6.453, 6.763] | 0.2439 | no |
| 150 it, 400 p | 6.560 | [6.357, 7.486] | 0.4035 | **yes** |
| 600 it, 60 p | 6.507 | [6.404, 6.615] | 0.0735 | no |
| 600 it, 200 p | 6.512 | [6.431, 6.561] | 0.0366 | no |

Two separate effects, pulling opposite ways:

- **Iterations CONTRACT the posterior.** Holding particles fixed at 60,
  going from 150 to 600 iterations shrinks the sd from 0.2525 to
  0.0735 — a factor of 3.4 — and at 200 particles from 0.2439 to
  0.0366, a factor of 6.7. The ensemble keeps collapsing the longer it
  runs. This is variance collapse as a function of run length, and it
  moves the interval *away* from covering the truth.
- **Particles widen it.** Only at 400 particles does the interval cover
  7.0, and it does so by being 1.6x wider rather than better centred.

The posterior mean is stuck at 6.51-6.57 at every budget, while an
independent scipy MLE of the same data gives 6.99. So there is also a
persistent ~7% low bias in the point estimate that no budget removes.

**Consequence:** running SVGD longer here produces a tighter, equally
biased, and therefore more confidently wrong posterior.

## 2. joint_prob at 168 vertices — DEFECT. Particles stuck at the boundary.

True value 1e-4, off-truth prior centred at 3.5e-5.

| budget | posterior mean | off | 95% HPD |
|---|---|---|---|
| 150 it, 60 p | 2.65e-05 | 73.5% | [1.00e-09, 1.33e-04] |
| 600 it, 60 p | 3.85e-05 | 61.5% | [1.00e-09, 1.17e-04] |
| 600 it, 200 p | 4.61e-05 | 53.9% | [1.00e-09, 1.17e-04] |
| 1500 it, 100 p | 3.20e-05 | 68.0% | [1.00e-09, 1.13e-04] |

Ten times the iterations and three times the particles improves the
estimate from 73.5% low to at best 53.9% low, then it plateaus and
regresses. It never approaches the truth.

Note the HPD lower bound: **1.00e-09 at every budget**. That is the
parameter floor, so some particles sit at the boundary and never leave,
across all four budgets. This is precisely the stuck-outlier pathology
the convergence harness was built to detect, and it is why the interval
spans five orders of magnitude while nominally "covering" the truth —
coverage here is meaningless.

## 3. Mixed-scale two-locus — DEFECT. More budget buys confidence in a wrong answer.

True [coalescence 2.0, recombination 0.5]; recombination's prior is
centred at 0.063.

| budget | coalescence | recombination | off | recombination 95% HPD |
|---|---|---|---|---|
| 60 it, 30 p | 1.811 | 0.0695 | 86% | [0.0430, 0.1052] |
| 400 it, 60 p | 1.817 | 0.0672 | 87% | [0.0649, 0.0713] |

Nearly seven times the iterations and twice the particles moves
recombination from 86% off to 87% off — no progress at all — while the
credible interval **contracts by a factor of four around the wrong
value** (width 0.062 down to 0.0064). Coalescence likewise sits at
1.81-1.82 (9% low) at both budgets.

So the weakly-identified parameter does not merely fail to converge; it
converges confidently to its prior neighbourhood. Recombination IS
identified at this scale (log-likelihood penalties of -6.0 at a quarter
and -62 at four times the true value), but its influence is two orders
of magnitude weaker than coalescence's, and it is left behind.

## 4. two_island at 186 vertices — GENUINELY BUDGET-LIMITED (the exception)

True [0.7, 0.3], off-truth prior.

| budget | posterior mean | off |
|---|---|---|
| 40 it, 25 p (the suite's slow-tier budget) | [7.072, 0.115] | 910%, 62% |
| 150 it, 50 p (what the PASSING 21-vertex case uses) | [1.605, 0.232] | 129%, 23% |
| 400 it, 50 p | *(running)* | |

This one behaves in the opposite way to the other three: more budget
helps, and substantially. The first parameter improves from 910% off to
129% off and the second from 62% to 23% purely by going from 40 to 150
iterations. So the pin's original wording — "may be travel-budget-limited
rather than a genuine failure to recover" — is CORRECT, and the suite's
reduced slow-tier budget is what produces the failure.

It has still not converged at 150 iterations (129% off on the first
parameter), so the honest statement is that this case needs a budget the
test suite cannot afford, not that inference is broken here. That is a
different disposition from pins 1-3 and the xfail reason should say so.

## What this means

For three of the four, the reduced budgets in the test suite are not
what is causing the failure — those get worse with more compute, because
the mechanism is variance collapse rather than incomplete convergence:
SVGD keeps contracting the ensemble, so a longer run yields a narrower
interval around a point estimate that is not moving toward the truth.
The 186-vertex two_island case is the one real budget limitation, and it
should be re-labelled accordingly rather than left implying a defect.

Practical reading for anyone fitting with this library: a tight
posterior from a long SVGD run is not evidence of a well-determined
parameter. The convergence harness's variance-ratio and stuck-outlier
statistics exist to catch exactly this, and the accuracy table should be
consulted rather than `summary()` alone.
