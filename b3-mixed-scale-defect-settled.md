# The mixed-scale FD defect, settled — and the larger error it was hiding

Settled 2026-08-16 against CLOSED FORMS (a defect in finite differences
cannot be adjudicated with finite differences). Artifact:
experiments/dr_fd_defect_settle.py, plus two diagnostics reproduced
below. Model: 2-phase hypoexponential, whose density and gradient are
analytic.

## Verdict in one line

The mixed-scale FD defect is real but MILDER and NARROWER than the
record claimed — and it sits on top of a much larger, scale-independent
error that nobody had measured: the forward PDF itself is only accurate
to about 0.25%, and that error passes straight through to the estimate.

## 1. What the record claimed

CLAUDE.md and memory state: every SVGD-facing gradient is a central
difference with an ABSOLUTE eps=1e-7 applied to every parameter, giving
"4-9% error at theta=[1,1e-8]". This was produced by inspection plus a
single pinned test; it was never swept against an independent oracle,
and never connected to whether an inference RESULT changes.

## 2. What is actually true (closed-form sweep, lam1 = 1.0)

| lam2 | scale ratio | forward value rel err | worst gradient rel err |
|---|---|---|---|
| 5.0e-01 | 2e0 | 2.86e-04 | 9.70e-04 |
| 1.0e-02 | 1e2 | 1.75e-04 | 8.18e-04 |
| 1.0e-04 | 1e4 | 9.40e-05 | 8.12e-04 |
| 1.0e-06 | 1e6 | 4.91e-05 | 5.36e-04 |
| 1.0e-07 | 1e7 | 7.45e-05 | 1.96e-03 |
| 1.0e-08 | 1e8 | 1.08e-03 | 2.61e-02 |
| 1.0e-09 | 1e9 | 8.89e-03 | 2.06e-01 |
| 1.0e-10 | 1e10 | 4.54e-02 | 6.91e-01 |

Findings:

- At theta=[1, 1e-8] the gradient error is **2.6%**, not the 4-9% on
  record. The claim was overstated for that point.
- The gradient does not exceed 10% error until a scale ratio of about
  **1e9**, i.e. until a parameter falls to ~1e-9 in absolute terms.
  Below 1e-7 the absolute probe crosses zero, which is where the daisy
  path's hard crash comes from (separately confirmed: the FFI refuses a
  negative rate).
- **There is an error FLOOR of roughly 1e-3 on the gradient at EVERY
  scale, including entirely benign ones.** The floor is not caused by
  finite differencing.

## 3. The floor: the forward PDF is only accurate to ~0.25% by default

Measured directly against the closed form at benign theta=(1.0, 0.5):

| granularity | max rel err of pdf |
|---|---|
| auto (= 1024) | 2.53e-03 |
| 512 | 5.06e-03 |
| 1024 | 2.53e-03 |
| 4096 | 6.31e-04 |
| 16384 | 1.58e-04 |
| 65536 | 3.94e-05 |
| 262144 | 9.85e-06 |

The error is first-order in 1/granularity, and the default is auto,
which resolves to 2*max(512, max_exit_rate) = 1024 for ordinary rates
(src/c/phasic.c, granularity derivation; the PDF call itself hardcodes
granularity=0 at src/cpp/parameterized/graph_builder.cpp:890, so a
caller of the default inference path cannot change it).

So the likelihood that every default continuous fit optimises is
accurate to about three digits.

## 4. That error is an inference BIAS, not just noise

Simulated 4,000 hypoexponential observations from theta=(1.0, 0.35) and
maximised two likelihoods on the SAME data — the closed form, and
phasic's — then compared the estimates:

| likelihood used | MLE | relative shift vs closed-form MLE |
|---|---|---|
| closed form | (1.07867031, 0.35140932) | — |
| phasic, granularity auto (1024) | (1.08140240, 0.35118075) | 2.53e-03 |
| phasic, granularity 16384 | (1.07883749, 0.35139534) | 1.55e-04 |
| phasic, granularity 262144 | (1.07868124, 0.35140830) | 1.01e-05 |

The shift tracks the PDF error term-for-term (2.53e-3 → 2.53e-3;
1.58e-4 → 1.55e-4; 9.85e-6 → 1.01e-5). It is therefore a systematic
bias in the estimator, not sampling noise.

Critically, this bias **does not shrink with more data**. At N=4,000 the
statistical error here is a few percent, so the bias is invisible. At
large N the statistical error falls below 0.25% and the bias becomes
the dominant error in the answer.

## 5. Consequences for the programme

1. **For the default continuous-PDF path, the accuracy limit is the
   FORWARD, not the gradient.** Making the gradient exact while the
   function being differentiated is 0.25% wrong cannot improve an
   estimate. This inverts the premise the whole exact-gradient
   programme was built on — for this path.
2. **The mixed-scale defect is real but narrow.** It needs a parameter
   below roughly 1e-7 absolute before it dominates the floor. Whether
   that regime occurs depends on parameterisation (a per-generation
   mutation rate of 1e-8 would sit squarely in it; a rescaled
   theta = 4*N*mu would not).
3. **Deferred 3 becomes the highest-value unit, for a reason nobody
   had noticed.** Its selected route — the Poisson mixture at pinned
   lambda — was measured during de-risk at 1.63e-11 value accuracy
   against the same closed forms, i.e. eight orders better than the
   current stepping. It would fix the forward bias AND supply an exact
   gradient in one change, on the quantity the default likelihood
   actually consumes.
4. **A cheaper interim exists**: raising the granularity used by the
   default PDF path (or exposing it) buys a decade of accuracy per 10x
   cost, with no new mathematics. It is a straight accuracy/time trade
   and should be measured before being chosen.

## 6. Correctness tests added

Per the user's instruction that correctness for any model is the
requirement and must be tracked by the suite, these measurements are
promoted to tests/pytest/test_likelihood_correctness.py — closed-form
accuracy of the PDF, the estimator-bias check, and the gradient sweep
across parameter scales. Known gaps are recorded as strict-xfail so the
suite states the aspiration and fails loudly if a gap is silently
closed or widened.
