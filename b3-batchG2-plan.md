# Batch G.2 — `exact_moment_grad` on the 2-D/multivariate rewards leaf + the §16b item-10 forward fix (plan v1)

**Master plan:** §9 (Batch G, last open leaf) + §16b item 10. **Origin of
scope:** Batch A's fold decision deferred the 2-D/multivariate kwarg
semantics ("not defined yet — G.2's design question"); Batch A also
ledgered the pre-existing 2-D-forward shape defect (§16b item 10) with
G.2 named as its natural vehicle. **USER DECISION (2026-08-15, recorded
here): FULL SYMMETRY** — the kwarg means the same thing on every moments
leaf; the multivariate wrapper gains it, svgd forwards it, R29's 2-D
rejection arm is retired. **Process:** `b3-execution-process.md` (two
plan refuters before code; the ladder G0-G5; both G3 amendments).
**Base:** master `658d36d0` (Batch C close-out; ninth ledger stamp
1992/0/84/24 — its environment caveat CLEARED by measurement 2026-08-15:
the exposure test + the full scc-parallelism file ran 5 passed in 6:50
on the awake machine; record the dated ledger note at G5).
**Branch:** `b3/batchG2-multivariate` — pure-Python batch (no C edits),
so per process §3.3 it MAY work in-place on the main checkout with
`pixi run install-dev` after every edit; a worktree is unnecessary.
G1's regression evidence is test-level (the A/B/C micro-gates don't
apply: the C core is untouched — state this in G1).

## 1. Scope

- **In:**
  1. `pmf_and_moments_from_graph_multivariate` gains
     `exact_moment_grad: bool = True` (the 1-D leaf's exact signature
     and default), forwarded VERBATIM to every per-feature
     `pmf_and_moments_from_graph` call — the missing opt-out ships.
  2. svgd: R29's 2-D arm is RETIRED (the rejection deleted); the svgd
     2-D-rewards leaf forwards an explicit `exact_moment_grad` to the
     multivariate wrapper (None = not forwarded, the D.4/A pattern),
     and `effective_options` reports it (status user/default as
     elsewhere).
  3. **§16b item 10 fix:** the 1-D `pmf_and_moments_from_graph` model
     REJECTS rewards with ndim >= 2 LOUDLY at the existing
     `_check_rewards_len` static-shape guard (a `ValueError` naming
     `pmf_and_moments_from_graph_multivariate` as the route), replacing
     today's obscure XLA shape-contract crash
     ("Expected: (2, 4), Actual: (2, 2)"). No legitimate caller
     breaks: the multivariate wrapper slices per-feature 1-D before
     calling the 1-D model, and svgd's 2-D leaf routes to the wrapper;
     direct 2-D-on-1-D usage CANNOT work today (it crashes) — verified
     pre-A, `b3-batchA-findings.md`.
- **Out / untouched:** the C core and all wrappers (no native edits);
  `mcmc` (its multivariate call site inherits the new default-True
  kwarg — same behavior as today; one findings line); the joint-index/
  daisy/epoch leaves (their kwargs are `exact_grad`/`exact_final_grad`,
  already shipped); `moments_from_graph`/`method_of_moments` (§16b
  item 5, unchanged).

## 2. Changes (all Python)

1. **`pmf_and_moments_from_graph_multivariate`**
   (`src/phasic/__init__.py:9183`): signature `+ exact_moment_grad:
   bool = True`; docstring gains the parameter (pointing at the 1-D
   docstring's decline ladder — per-feature models own the
   static/dynamic declines and their INFO logs, which therefore fire
   PER FEATURE); every internal `pmf_and_moments_from_graph(...)` call
   gains `exact_moment_grad=exact_moment_grad` (GROUNDING at
   implementation: enumerate ALL internal build sites in the wrapper —
   expected one per feature loop + possibly a shared/1-D fallback
   path; every one must forward, none may be missed — the
   "silently-inert" defect class).
2. **`svgd_config.py`:** delete R29's `rewards_kind == '2d'` raise;
   rewrite the R29 leading comment (fourth rewrite: the kwarg is now
   honored on ALL moments leaves — no-rewards, 1-D, 2-D/multivariate;
   the remaining R29 rejections are epoch_starts and joint-prob kinds
   only; builder-level static declines stay accepted-but-INFO-logged
   per the A G4 disposition).
3. **svgd 2-D leaf** (`__init__.py:6511` region): the `_emg_kw`-style
   conditional splat (None ⇒ absent; the A pattern at the 1-D leaf) on
   the `pmf_and_moments_from_graph_multivariate(...)` call.
   GROUNDING: confirm `exact_moment_grad` reaches `from_svgd_call` and
   the effective-options ledger already handles it (it does for the
   1-D leaf — R29 relaxation is config-only).
4. **`_check_rewards_len`** (the 1-D model's static-shape guard):
   reject `ndim >= 2` with the actionable ValueError. The backward's
   2-D static-decline arm (`_rewards_1d` dispatch) becomes unreachable
   — KEEP as defense-in-depth with its comment updated to say so (the
   A-era INFO-decline text is retired from the reachable surface).
5. **Shipped text:** svgd docstring's `exact_moment_grad` section (the
   R29 clause: 2-D no longer rejected — honored via the multivariate
   wrapper); pmf_and_moments docstring (the "2-D rewards keep FD on
   this leaf" cause is replaced by the loud rejection; the
   multivariate pointer stays); R29 comment (item 2); CLAUDE.md at G5
   (the "multivariate has no passthrough kwarg" gap paragraph +
   §16b item 10 CLOSED + the A-disposition exception text's 2-D
   mention); memory at G5.

## 3. Test plan (fate table + new cells)

**Fate flips (both are rejection pins on the arm being retired):**
- `inference/test_svgd_exact_moment_grad_kwarg.py::TestR29...::
  test_rewards_2d_rejected` → `test_rewards_2d_honored` (fit
  completes; effective_options status 'user'; mirrors the A-batch 1-D
  flip precedent exactly).
- `inference/test_svgd_exact_moment_grad_rewards.py::
  test_rewards_2d_still_rejected_with_g2_message` → reworked to the
  honored contract (the "G.2 message" it pins is being retired).
- GROUNDING at implementation: grep tests for the R29-2-D message
  fragments ("2-D (multivariate) rewards", "Drop exact_moment_grad")
  and for constructions relying on the XLA crash — enumerate every
  hit in the fate table before editing.

**New cells (`inference/test_svgd_exact_moment_grad_kwarg.py` gains the
svgd cells; a new `inference/test_exact_grad_multivariate_kwarg.py`
holds the wrapper-level cells):**
1. Wrapper forwarding-discrimination (spy on
   `Graph.pmf_and_moments_from_graph`): explicit False arrives at
   EVERY per-feature call; default (omitted) arrives as True — the
   signature default, NOT absent (differs from svgd's None-absent
   convention; assert exactly).
2. Wrapper opt-out changes behavior: 2-D model with
   `exact_moment_grad=False` → zero `_moments_grad_theta` spy calls
   under grad; True/default → ≥ n_features full-size successes
   (the B/C success-floor discipline).
3. Wrapper value parity: gradient with default vs explicit True
   BITWISE; False vs default parity vs central-diff (both correct,
   different paths — measured actuals at implementation).
4. svgd front door: `svgd(obs2d, rewards=2d, exact_moment_grad=False)`
   completes finite, effective_options user/False, spy shows zero
   exact calls; True completes with ≥ n_particles successes; None →
   default behavior byte-identical to today (golden or spy-based).
5. §16b item 10: the 1-D model with 2-D rewards raises the actionable
   ValueError at call (grad AND forward), naming the multivariate
   route; the multivariate wrapper with the same rewards WORKS
   (regression-guarded by the existing A/B/C multivariate cells).
6. R29 comment/message coherence: no svgd_config path mentions the
   retired 2-D rejection (grep-cell or review-verified).

## 4. De-risks (minimal — pure-Python, all patterns proven)

- **D-G2.1:** enumerate the wrapper's internal build sites (§2.1
  grounding) + probe the forwarded-False composition on a 2-feature
  model (spy: zero exact calls) BEFORE editing svgd — pins the seam.
- **D-G2.2:** pin today's XLA crash text on the 1-D-model 2-D-rewards
  route (the §16b-10 baseline) and verify `_check_rewards_len` is
  reached under grad AND forward for the rejection (it runs at the top
  of `_compute_pure`, which the custom_vjp fwd/bwd share — verify the
  bwd FD probes also pass through it, else the error could surface
  from a weirder place).
- D-G2.3 (only if D-G2.1 surfaces >1 build-path): extend the spy to
  every path.

## 5. Gates

- **G0:** ninth stamp current at `658d36d0`; the caveat-clearing
  measurement recorded (this plan's header); delta above the stamp =
  this plan commit (docs-only).
- **G1:** the new test file + the two flipped files green; D-G2 probes
  ALL PASS. (No C micro-gates: the native layer is untouched — the
  A/B/C gates would be vacuously unchanged; G2's map covers regression
  via the existing multivariate/rewards suites.)
- **G2 (map):** svgd-config row + moments-rewards row + B/C rows'
  pytest files (`test_exact_grad_{rewards,formula_mode,callback_mode,
  discrete,log_weight_mode}.py`) + `test_multivariate_correctness.py`
  + `test_fd_gradient_mixed_scale.py` + `test_svgd_config.py`.
- **G3:** chunked vs the ninth stamp (1992/0/84/24), both amendments
  (groups enumerated from split output, union == collected, output per
  group, `-rf`, preserved outputs).
- **G4:** two adversarial diff refuters (attack: the forwarding seam's
  completeness across ALL wrapper build sites; the default-True vs
  None-absent convention difference between wrapper and svgd; the
  retired-arm text sweep; the ValueError's reachability under
  vmap/jit tracing — a static-shape guard must raise at TRACE time,
  verify it does; fate-table completeness).
- **G5:** merge review; squash-merge (or direct-commit flow if worked
  in-place — the branch still exists for review lineage); 10th ledger
  stamp measured (expect 1992 + N_new); tracker (G.2 → merged; BATCH G
  fully closed; Phase 4 COMPLETE — the last G leaf); master plan §9
  tick + §16b item 10 CLOSED; CLAUDE.md; process-map row for the new
  test file; memory; install rebuild.

## 6. Ledger arithmetic

Expected G3 = 1992 + N_new (≈6-8 cells minus the two flips' net; pinned
at G1). Both flipped files keep their counts (rework, not delete).

## 7. Risks / open questions carried into review

1. The wrapper's default is `True` (a real default) while svgd's is
   `None` (not forwarded): after this batch the svgd-None route and
   the wrapper-default route must be BEHAVIORALLY identical (both =
   per-feature default True). A refuter should probe the three svgd
   states (None/True/False) end-to-end.
2. The retired R29 arm's message is load-bearing in two shipped tests
   (the fate flips) — and possibly in docs/comments beyond the §2.5
   list; the sweep is a refuter mandate.
3. `_check_rewards_len` raising on ndim>=2: confirm no OTHER model
   family reuses this helper with legitimate 2-D rewards (grep its
   call sites — if shared, scope the rejection to the 1-D leaf's
   instance).
4. mcmc's multivariate call site: inherits default-True — identical to
   today's behavior; confirm no mcmc kwarg plumbing is implied (none
   planned; mcmc has no exact_moment_grad kwarg — a findings line).

## v2 amendment (2026-08-15, post two-refuter review — BINDING over v1)

**Verdicts:** design/seams SOUND-WITH-CORRECTIONS; completeness/process
SOUND-WITH-CORRECTIONS. Convergent top finding re-presented to the user.

### A. USER DECISION 2 (2026-08-15): UNIFORM rejection of 2-D rewards on the 1-D model

v1's premise "direct 2-D-on-1-D always crashes" was REFUTED by both
refuters' probes: only the DEFAULT (pybind) path crashes (real text:
`Incorrect output shape for return value #0: Expected: (4, 5), Actual:
(4, 2)` — root cause: the shape spec reads `rewards.shape[1]` = n_vertices
where the feature count belongs, `__init__.py:8026`); the FFI path
returns SILENT GARBAGE (correct values flattened into row 0, remaining
feature rows zero); the CALLBACK path (Batch C's shape contract uses
`shape[0]`, the right axis) accidentally WORKS and is probe-verified
VALUE-IDENTICAL to the multivariate wrapper. **Decision: reject
uniformly on all three paths** — the shared `_check_rewards_len`
(defined once, `__init__.py:7730`, closed over by all three
`_compute_pure` variants — ONE edit site) raises the actionable
ValueError (stating the CORRECT orientation `(n_features, n_vertices)`
and naming `pmf_and_moments_from_graph_multivariate`). This fixes the
crash AND the silent-garbage path, and knowingly retires the
undocumented accidental callback capability — migration is lossless
(value parity probed). Recorded as a deliberate break, not a side
effect.

### B. svgd cells run on SparseObservations; NEW additive rule R32

The pinned dense-obs constructions can NEVER complete with 2-D rewards
(dense 2-D obs raises "use dense_to_sparse()" at `__init__.py:6274`;
dense 1-D obs + 2-D rewards dies deep in `_log_lik_from_pmf`,
`svgd.py:6068`) — the ONLY completing route is SparseObservations
(probed end-to-end; ledger row default/None). Therefore: (1) the
honored-contract cells (fate flips + cell 4) use `dense_to_sparse`;
(2) NEW additive rule **R32**: dense observations + 2-D rewards →
crisp actionable SvgdConfigError naming `dense_to_sparse` (an
error-experience improvement over today's deep shape errors for BOTH
kwarg and kwarg-less routes; the retired R29 arm's crispness is thereby
retained and improved for the dense construction).

### C. Architecture corrections (both refuters)

The wrapper has exactly ONE internal build site (`model_1d =
pmf_and_moments_from_graph(...)` at `__init__.py:9275`, WRAPPER-call
time; dense/sparse/1-D/None paths all reuse the closure; spy-probed:
1 build call, 0 per model call). Forwarding = one kwarg on one call.
Test cell 1 reworded: the single wrapper-build call receives the
value (runtime per-feature discrimination stays in cell 2 via the
`_moments_grad_theta` spy ≥ n_features successes). Construction-time
static declines fire ONCE (not per feature); only backward per-theta
declines are per-feature — §2.1's log claim corrected.

### D. Shipped-text sweep additions (completeness M2 + design M5)

Beyond v1 §2.5: `svgd_config.py:288-291` field comment ("only honored
on the no-rewards moments model" — stale since A); **R31's live message
`svgd_config.py:1292-1294`** ("exact_moment_grad ... unavailable on
rewards-bearing models until the rewards adjoint ships" — stale since
A, wrong post-G.2: REWORD); CLAUDE.md's B3-section sentence "A direct
2-D-rewards call on the 1-D leaf fails in the FORWARD..." (now: loudly
rejected by decision); the svgd docstring lead-in `:5730-5734` (2-D
joins the honored list) AND the 2-D orientation error at `:5820-5824`
("(n_vertices, n_features)" contradicts the validator/wrapper — fix to
`(n_features, n_vertices)`); module docstrings of the three touched
test files + the multivariate engagement test's docstring (the
"pre-existing shape-contract defect" narrative gets its closure note).

### E. Scope/process corrections

- `method_of_moments.py:389` is the third wrapper consumer —
  behavior-neutral (inherits default True; scipy does its own FD);
  findings line alongside mcmc.
- "Direct-commit flow" STRUCK (process §5.3): work in-place is
  permitted (§3.3) but all commits land on `b3/batchG2-multivariate`,
  then squash-merge from master.
- G2 map += `test_multivariate.py`, `test_multivariate_length1.py`,
  `test_notebook_multivar_reproduction.py`.
- NEW cell: exposure × 2-D rewards × explicit kwarg (previously killed
  by the retired arm; now accepted with R11's UserWarning — assert the
  warn fires and the fit completes).
- G0 wording: two docs-only commits above the ninth stamp
  (`7e1e162c` incl. the ALREADY-RECORDED ledger re-verification note —
  no second recording at G5).
- G5 += tracker G-row staleness fix ("Leaves 3/4 still blocked(A)" —
  stale since A delivered leaf 3).

### F. De-risks re-scoped

D-G2.2 is RESOLVED at review (all three paths probed and pinned:
crash text, garbage shape, callback parity; trace-time raise legibility
proven on the existing guard under jit/vmap compositions). D-G2.1
(forwarded-False ⇒ zero exact calls, 2-feature spy probe) remains the
one pre-implementation check. Rule-enumeration also resolved:
`rewards_kind` consumers are exactly R3/R5/R11/R29; no rule depended on
the retired arm.
