# Scale reality check: do the shipped exact gradients work on TYPICAL models?

**Trigger (2026-08-16, user):** typical production models are
**20,000-60,000 vertices**, and phasic must also handle **~10x that
(200,000-600,000)**. This was not recorded anywhere in the project docs
(now fixed: `CLAUDE.md` "Scale targets" + memory
`project_model_scale_targets`). Every prior B3 measurement was taken on
toy fixtures or at n<=8,407 — so the program's real-world coverage had
never been checked against the actual target range.

**Artifacts:** `experiments/dr_scale_gradient_ceiling.py` (memory-capped
ladder), plus two diagnostics recorded inline below.

## Headline: on the user's real model class, the shipped exact-moments gradient does NOT run — for TWO independent reasons, one cheap to fix and one expensive

## 1. The measured ceiling (two-locus ARG ladder, memory-capped)

Child processes with a parent RSS watchdog at 6.5 GB, 900 s time-box,
staged sizes, ladder aborted on any wall:

| family | nr | vertices | op time | peak RSS | exact path used? |
|---|---|---|---|---|---|
| moments | 6 | 1,044 | 0.07 s | 0.43 GB | **NO — declined** |
| moments | 8 | 8,407 | 97.5 s | 5.11 GB | **NO — declined** |
| moments | 9 | 22,653 | **TIME-WALL @900 s** | 4.96 GB | — |
| sojourn | 6 | 1,044 | 0.07 s | 0.42 GB | yes (gate skipped) |
| sojourn | 8 | 8,407 | 94.0 s | 5.88 GB | **NO — declined** |
| sojourn | 9 | 22,653 | **TIME-WALL @900 s** | 5.32 GB | — |

**Reading:** 22,653 vertices — the LOW end of the typical range — does
not complete a single gradient call within 15 minutes. The 200k-600k
target is not within reach of the monolithic path by any margin. Note
the ~95 s at 8,407 vertices is spent building the tape *and then
declining* — i.e. paid in full for an FD answer.

## 2. The second blocker, found en route: the conditioning gate declines on real two-locus graphs

At nr=6 (1,044 vertices) and a perfectly benign `theta=[2.0, 5.0]`:

```
moments J.size = 0                      -> DECLINED
moments @ PHASIC_CONDITION_THRESHOLD=1e300 -> size 2, ACCEPTED
sojourn(nogate) len = 8                 -> accepted
sojourn(gated)  len = 0                 -> DECLINED
```

So the decline is **the MPFR conditioning gate**, not scale, not
`was_dph`, not topology. And the answer the gate refuses to give is
correct: against a Richardson relative-step reference on the same
graph,

```
gate-LIFTED exact J : [-0.7540948993120581, 0.04147829578962932]
Richardson reference: [-0.7540948993126998, 0.041478295789376673]
max rel diff        : 8.5e-13   -> the lifted answer is GOOD
```

**Consequence:** with `exact_moment_grad=True` (the shipped default), a
user fitting a two-locus model gets **finite differences**, logged at
INFO, after paying the full exact-path setup cost. The exact gradient
the program was built to deliver is not delivered on this model class
at all.

### Why CC-2 / Deferred-4 did not catch this

The Phase-0 sweep (`b3-d4-sweep-findings.md`, PARKED 2026-08-15 with
user sign-off) verified the moments gradients are "never silently
wrong" and found the gate "3-4 decades EARLY" — but it swept **toy
fixtures** (chain2-class), where conservatism is harmless because the
gate does not actually bite. It never swept a production-class
two-locus graph, where the gate bites at benign theta. The park
decision was therefore made on unrepresentative fixtures. This is new
evidence that reopens Deferred 4's *cheap* end — NOT the full MPFR
adjoint (Phases 1/2, still unjustified), but a gate opt-out.

**Precedent already exists in the tree:** Batch H hit exactly this for
the sojourn family — "the gate declines 100% of realistic
coalescent-scale calls while its lifted answers match an fp64 oracle to
~1e-13" — and shipped `ptd_sojourn_grad_theta_subset_nogate` /
`skip_condition_gate=True` as an additive opt-out (user-decided
2026-08-13). The moments family has no equivalent. The measurement
above is the moments-family twin of that finding, on the same kind of
model, with the same character (lifted answer good to ~1e-12/1e-13).

## 3. Platform note (corrects the D1 plan's I4 protocol)

The adversarial review of `b3-d1-implementation-plan.md` required
child-side `resource.setrlimit(RLIMIT_DATA/RLIMIT_AS)` plus a parent
RSS watchdog. **Measured: macOS rejects lowering both limits on this
machine** — `setrlimit` raises `ValueError: current limit exceeds
maximum limit` for `(cap, cap)` and `(cap, INF)` alike, even though
`getrlimit` reports both as INFINITY. So on this platform the
**parent-side RSS watchdog is the ONLY working memory defence**, and it
must not be described as belt-and-braces. The harness above degrades to
watchdog-only (best-effort rlimit, exception swallowed) and that is what
actually protected the two TIME-WALL cells.

## 4. What this means for the three pending decisions

- **Deferred 1 (hierarchical/SCC adjoint)** — its activation premise
  moves from "a forcing model exists" to "the ordinary case is out of
  reach". Its plan's scale gates, drafted at nr=8 (8,407 vertices), are
  UNDERSIZED: parity must still be proven where monolithic works
  (nr<=8), but FEASIBILITY must be demonstrated across 20k-60k and a
  measured path shown toward 200k-600k.
- **A new, cheap, independent unit** is now visible: a moments-family
  conditioning-gate opt-out mirroring the shipped sojourn one. It is
  small, has direct precedent, and unblocks exact gradients on real
  models at sizes where the monolithic path still completes — i.e. it
  pays off immediately and regardless of whether Deferred 1 is built.
- **Deferred 2 / 3** are unaffected in kind (both concern different
  quantities), but both inherit the same scale question for their own
  implementations, and D2's own cost model already concluded that
  RECOMPUTE beats RETAIN — the same lesson Deferred 1's retention
  design now has to face at 200k-600k vertices.
