# Plan — consistent reward-length handling (and the discrete-graph inconsistency behind it)

Branch context: fixes for N1/N2/variance_discrete/moments_discrete already landed on
`fix/is-discrete-propagation`. This plan covers the remaining N4 reward-length issue and the
related `was_dph` normalisation inconsistency it exposed. **Not yet implemented — awaiting the
design decision in §4.**

## 1. The finding (grounded, not assumed)

The reward-length convention is **not** ambiguous. The canonical length is **`vertices_length()`**,
enforced everywhere except the parameterized path:

| path | reward length it accepts | enforced? |
| --- | --- | --- |
| `g.reward_transform(r)` (pure Python) | `== vertices_length()` | **yes** — ValueError otherwise |
| `g.expectation(rewards=r)` / `_validate_rewards` | `== vertices_length()` | **yes** |
| `discretize()` → `graph.rewards` | `len == vertices_length()` (incl. start + aux) | it *is* the reference |
| C `ptd_graph_reward_transform` | reads `rewards[i]` for `i < vertices_length` | needs exactly that many |
| **`pmf_and_moments_from_graph` (+ FFI)** | **anything** | **NO** — the bug |

Evidence: for a 3-vertex graph (states `[[0],[1],[2]]`, start included) `vertices_length()==3`;
`reward_transform`/`expectation` **raise** on length 2 and accept length 3;
`ptd_graph_reward_transform` (`phasic.c:6659`) loops `i < vertices_length`.

Consequences on the parameterized path:
- **too short** → out-of-bounds heap read (non-deterministic across processes — the N4 finding).
- **too long** → silently truncated (extra entries ignored).
- Existing tests `inference/test_rewards_support.py` and `inference/test_multivariate.py` pass
  length `vertices − 1` ("one per vertex excluding start") — a **misconception**; those exact
  lengths are rejected by the canonical `reward_transform`. They only "pass" because the OOB read
  of the missing entry happens not to crash and the assertions are loose.

So this is an **unenforced convention** + **tests/docs that encode a wrong belief**, not a genuine
two-convention split. My earlier N4 guard used the correct count (`vertices_length()`); reverting it
was right only because it would have surfaced the mis-lengthed tests without fixing them.

## 2. Proposed fix (consistent + robust + fail-loud)

- **Enforce the canonical length on the parameterized path.** In `pmf_and_moments_from_graph` (and
  `_multivariate`), validate the reward shape against `vertices_length()` before dispatch, reusing
  the existing `_validate_rewards` so the count is definitionally correct and the error message
  matches the rest of the API. Prefer a **Python-level static-shape check** at model-construction /
  first-call (jit-safe: shapes are static) so the user gets a clean `ValueError`, not an
  `XlaRuntimeError` from a C++ throw inside a callback.
- **Re-add the C++ guard** as a backstop (validate against `n_vertices_`, the serialized count =
  `vertices_length()`), so no path can OOB even if a future caller bypasses the Python layer.
- **Fix the mis-lengthed tests + docs**: `test_rewards_support.py`, `test_multivariate.py` →
  length `vertices_length()`; correct every "excluding start" comment.

## 3. Related inconsistency to fix in the same pass: `was_dph` normalisation

Uncovered while binding `moments(discrete=True)`. A **native DPH** (`is_discrete` set, `was_dph`
NOT set) and a **`discretize()`'d graph** (`was_dph` set) behave differently under
`update_weights`: `was_dph` triggers auto-normalisation that rescales each vertex's out-edges to sum
to 1, **collapsing a native DPH to a deterministic walk** (`moments` then returns `[2,6,24]` instead
of the true DPH moments). Verified directly.

- `serialize()` currently carries `is_discrete` but **not** `was_dph`; `from_serialized` (my Batch 2
  change) latches `was_dph` whenever `is_discrete` — which would wrongly normalise a reconstructed
  native DPH.
- **Fix**: serialize **both** `is_discrete` and `was_dph`; `from_serialized` restores each faithfully
  (drop the unconditional `set_was_dph`). Then native-DPH and discretized graphs round-trip
  correctly and `update_weights` normalises iff the original graph did.

## 4. Design decision needed before implementing

`vertices_length()` counts **all** vertices including the start and the absorbing state, whose reward
entries are effectively unused (no accumulated sojourn). Two options for the canonical convention:

- **(A) Keep `vertices_length()`** — least disruptive (already the canonical count everywhere, matches
  `graph.rewards`); a couple of entries are "don't care". Enforce it uniformly.
- **(B) Move to "one reward per transient (non-absorbing) vertex"** — semantically cleaner UI (a
  reward exactly where sojourn accumulates), but a larger change: it redefines the length everywhere
  (`reward_transform`, `expectation`, `_validate_rewards`, `graph.rewards`) and touches every caller.

Recommendation: **(A)** now (consistency + safety with minimal churn), and consider (B) only as a
separate, deliberate UX change if you want the cleaner surface.

## 5. Batches (test-gated)

- **B1** — parameterized-path reward-length validation (Python static-shape check reusing
  `_validate_rewards` + C++ backstop). Fix mis-lengthed tests/docs. Re-add the N4 guard.
- **B2** — serialize + restore `was_dph` faithfully; drop the unconditional `set_was_dph` in
  `from_serialized`. Regression on discretize round-trips + native-DPH moments.
- **B3** (optional, only if §4 → B) — migrate to the transient-only reward convention.
